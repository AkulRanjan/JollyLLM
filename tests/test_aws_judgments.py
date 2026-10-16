from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from legal_graph.aws_judgments import (
    clean_pdf_text,
    extract_aws_years,
    official_label,
    parse_metadata,
    read_aws_records,
    select_aws_records,
)
from legal_graph.models import Split


def _raw_html(title: str, decided: str, disposal: str) -> str:
    petitioner, respondent = title.split(" versus ")
    return (
        f"<button type='button' aria-label=\" {title} pdf\" class='noToken'><font size='4'> <strong>{petitioner}"
        f"<span class='fst-italic'> versus </span>{respondent} </strong>- <span class='escrText'>[2015] 10 S.C.R. 1013</span>"
        f"<span class='ncDisplay'>2015 INSC 797</span></button><strong class='caseDetailsTD' >"
        f"<span style='color:#212F3D' > Decision Date :</span><font color='green'> {decided}</font>"
        f"<span style='color:#212F3D' > | Case No :</span><font color='green'> CIVIL APPEAL No. 1/2015</font>"
        f"<span style='color:#212F3D' > | Disposal Nature :</span><font color='green'> {disposal}</font>   "
        f"<span style='color:#212F3D' > |  Bench :</span><font color='green'> 3 Judges</font></strong>"
    )


BODY = "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1 of 2015. " + "The facts are stated at length here. " * 40


def _row(path: str, decided: str, disposal: str, ending: str, **extra) -> dict:
    return {
        "bench": "3 Judges", "case_number": "CA 1/2015", "decision_date": decided, "disposal_nature": disposal,
        "metadata_sha256": hashlib.sha256(f"m{path}".encode()).hexdigest(), "neutral_citation": "",
        "page_count": 3, "partition_year": int(decided[:4]) if decided else 2015, "path": path,
        "pdf_sha256": hashlib.sha256(f"p{path}".encode()).hexdigest(), "scr_citation": "",
        "text": "A v. B\n" + BODY + "\n" + ending, "title": "A v. B", **extra,
    }


def _write_records(path: Path, rows: list[dict]) -> str:
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MetadataTests(unittest.TestCase):
    def test_parses_title_citations_and_case_details(self) -> None:
        fields = parse_metadata({"path": "2015_10_1013_1034",
                                 "raw_html": _raw_html("ALPHA &amp; ORS. versus BETA", "16-10-2015", "Appeal(s) allowed")})
        self.assertEqual(fields["title"], "ALPHA & ORS. v. BETA")
        self.assertEqual(fields["decision_date"], "2015-10-16")
        self.assertEqual(fields["disposal_nature"], "Appeal(s) allowed")
        self.assertEqual((fields["scr_citation"], fields["neutral_citation"]), ("[2015] 10 S.C.R. 1013", "2015 INSC 797"))
        self.assertEqual((fields["case_number"], fields["bench"]), ("CIVIL APPEAL No. 1/2015", "3 Judges"))

    def test_unparseable_date_is_left_empty(self) -> None:
        self.assertEqual(parse_metadata({"path": "x", "raw_html": _raw_html("A versus B", "soon", "Dismissed")})["decision_date"], "")

    def test_only_four_way_disposals_map_to_labels(self) -> None:
        self.assertEqual(official_label("Appeal(s)  allowed"), "allowed")
        self.assertEqual(official_label("Case Allowed"), "allowed")
        self.assertEqual(official_label("Dismissed"), "dismissed")
        self.assertEqual(official_label("Disposed off"), "disposed")
        self.assertEqual(official_label("Case Partly allowed"), "partly_allowed")
        for other in ("Matter referred to larger bench", "Hearing Adjourned", "Dismissed After", ""):
            self.assertIsNone(official_label(other), other)


class CleanTextTests(unittest.TestCase):
    def test_drops_margin_letters_page_numbers_and_running_headers(self) -> None:
        header = "1024 SUPREME COURT REPORTS [2015] 10 S.C.R."
        bodies = ("The appellant was dis-\ncharged from service and\nB\nfiled an appeal.", "The respondent opposed it.",
                  "Counsel argued at length.")
        pages = [f"{header}\nA\nALPHA v. BETA {1024 + n}\n{body}\n{1024 + n}" for n, body in enumerate(bodies)]
        text = clean_pdf_text(pages)
        self.assertNotIn("SUPREME COURT REPORTS", text)
        self.assertNotIn("ALPHA v. BETA", text)
        self.assertNotRegex(text, r"(?m)^\d+$")
        self.assertEqual(text.split("\n"), ["The appellant was discharged from service and filed an appeal.",
                                            "The respondent opposed it.", "Counsel argued at length."])

    def test_keeps_paragraph_breaks_and_headings(self) -> None:
        text = clean_pdf_text(["The appeal was heard.\n2. The facts are these.\nJUDGMENT\nWe have heard counsel."])
        self.assertEqual(text.split("\n"), ["The appeal was heard.", "2. The facts are these.", "JUDGMENT",
                                            "We have heard counsel."])


class SelectionTests(unittest.TestCase):
    def test_years_labels_and_overlap_decide_what_is_kept(self) -> None:
        rows = [
            _row("2013_1_1_9", "2013-03-01", "Appeal(s) allowed", "The appeal is allowed."),
            _row("2015_2_5_9", "2015-12-30", "Case Partly allowed", "The appeal is partly allowed."),
            _row("2017_3_1_9", "2017-05-02", "Dismissed", "The appeal is dismissed."),
            _row("2016_9_771_799", "2015-10-07", "Dismissed", "The appeal is dismissed."),  # also in SC-2016
            _row("2016_1_1_9", "2016-02-01", "Dismissed", "The appeal is dismissed."),  # the SC-2016 year
            _row("2014_1_1_9", "2014-01-01", "Dismissed", "The appeal is allowed."),  # text disagrees
            _row("2014_2_1_9", "2014-01-01", "Disposed off", "Ordered accordingly."),  # no phrase in the text
            _row("2014_3_1_9", "2014-01-01", "Hearing Adjourned", "The appeal is dismissed."),
            _row("2014_4_1_9", "", "Dismissed", "The appeal is dismissed."),
        ]
        rows.append(dict(rows[0], partition_year=2014))  # the same report listed under a second year
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl.gz"
            digest = _write_records(path, rows)
            records, report = select_aws_records(path, source_id="source:aws:test", expected_sha256=digest,
                                                 train_years=(2013, 2014, 2015), test_years=(2017,),
                                                 exclude_paths=("2016_9_771_799",))
        by_path = {record.attributes["aws_path"]: record for record in records}
        self.assertEqual(set(by_path), {"2013_1_1_9", "2015_2_5_9", "2017_3_1_9"})
        self.assertEqual(by_path["2013_1_1_9"].origin_split, Split.TRAIN)
        self.assertEqual(by_path["2015_2_5_9"].attributes["official_label"], "partly_allowed")
        self.assertEqual(by_path["2017_3_1_9"].origin_split, Split.TEST)
        self.assertTrue(by_path["2017_3_1_9"].case_id.startswith("case:india:sc:2017:sha256-"))
        self.assertEqual(report["kept"], {"test": 1, "train": 2})
        self.assertEqual(report["skipped"], {
            "duplicate_report": 1, "in_sc2016": 1, "no_decision_date": 1, "no_disposition_phrase_in_text": 1,
            "no_four_way_disposal": 1, "other_year": 1, "text_disagrees_with_official_label": 1,
        })

    def test_checksum_and_year_rules_are_enforced(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl.gz"
            _write_records(path, [_row("2013_1_1_9", "2013-03-01", "Dismissed", "The appeal is dismissed.")])
            with self.assertRaises(ValueError):
                select_aws_records(path, source_id="s", expected_sha256="0" * 64, train_years=(2013,), test_years=(2017,),
                                   exclude_paths=())
            for train, test in (((2013, 2016), (2017,)), ((2013,), (2013,))):
                with self.assertRaises(ValueError):
                    select_aws_records(path, source_id="s", expected_sha256=None, train_years=train, test_years=test,
                                       exclude_paths=())


@unittest.skipUnless(importlib.util.find_spec("pymupdf"), "pymupdf is not installed")
class ExtractionTests(unittest.TestCase):
    def test_extracts_each_year_from_its_archives(self) -> None:
        import pymupdf

        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "CIVIL APPELLATE JURISDICTION: the appeal is dismissed.")
        pdf = document.tobytes()
        metadata = json.dumps({"path": "2015_1_1_2", "raw_html": _raw_html("A versus B", "02-01-2015", "Dismissed")}).encode()

        def tar(members: dict[str, bytes]) -> bytes:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            return buffer.getvalue()

        bucket = {
            "data/tar/year=2015/english/english.index.json": json.dumps({"parts": [{"name": "english.tar"}]}).encode(),
            "data/tar/year=2015/english/english.tar": tar({"2015_1_1_2_EN.pdf": pdf, "2015_9_9_9_EN.pdf": pdf}),
            "metadata/tar/year=2015/metadata.index.json": json.dumps({"parts": [{"name": "metadata.tar"}]}).encode(),
            "metadata/tar/year=2015/metadata.tar": tar({"2015_1_1_2.json": metadata}),
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "out" / "records.jsonl.gz"
            summary = extract_aws_years([2015], Path(directory) / "work", output, base_url="bucket",
                                        opener=lambda url: io.BytesIO(bucket[url.removeprefix("bucket/")]))
            rows = read_aws_records(output)
            self.assertEqual(summary["record_count"], 1)
            self.assertEqual(summary["counts_by_year"]["2015"]["pdf_without_metadata"], 1)
            self.assertEqual(summary["records_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(len(summary["archives"]), 2)
            self.assertEqual(list((Path(directory) / "work").iterdir()), [])  # archives are deleted after use
        self.assertEqual((rows[0]["path"], rows[0]["decision_date"], rows[0]["disposal_nature"]),
                         ("2015_1_1_2", "2015-01-02", "Dismissed"))
        self.assertIn("the appeal is dismissed", rows[0]["text"])


if __name__ == "__main__":
    unittest.main()

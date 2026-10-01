import json

from whospoke.llm_postprocess import PostProcessor, StaticLLM, parse_json_object


def transcript():
    return {
        "order": "B",
        "duration_s": 6.0,
        "lines": [
            {"speaker": "Speaker_A", "start": 0.0, "end": 2.0,
             "text": "आई डी कार्ड गुम हो गया है सर", "hinglish": "aai D card gum ho gaya hai sir", "audio": "mixture"},
            {"speaker": "Speaker_B", "start": 2.1, "end": 4.0,
             "text": "आधार कार्ड की फोटो कॉपी लाना", "hinglish": "aadhar card ki photo copy lana", "audio": "mixture"},
        ],
    }


def model_outputs():
    chunk = {
        "cleaned_dialogue": [
            {"line_id": 1, "speaker": "FAKE", "start": 99, "end": 100,
             "text": "आई डी कार्ड गुम हो गया है।", "hinglish": "aai D card gum ho gaya hai sir",
             "translation": "My ID card has been lost, sir.", "uncertain": False, "note": ""},
            {"line_id": 2, "speaker": "Speaker_B", "start": 2.1, "end": 4.0,
             "text": "आधार कार्ड की फोटो कॉपी लाना।", "hinglish": "aadhar card ki photo copy lana",
             "translation": "Bring a photocopy of the Aadhaar card.", "uncertain": False, "note": ""},
            {"line_id": 999, "speaker": "Speaker_X", "start": 10, "end": 11,
             "text": "Invented line", "hinglish": "invented", "translation": "Invented.", "uncertain": False, "note": ""},
        ],
        "chunk_summary": "The speakers discuss a lost ID card and required documents.",
        "keywords": ["ID card", "आधार कार्ड"],
        "action_items": [{"owner": "Speaker_B", "action": "Bring a photocopy of the Aadhaar card.", "evidence_line_ids": [2]}],
    }
    synthesis = {
        "title": "ID Card and Required Documents",
        "topic": "ID card and required documents",
        "executive_summary": "The speakers discuss a lost ID card and bringing a photocopy of an Aadhaar card.",
        "key_points": ["An ID card has been lost.", "A photocopy of an Aadhaar card is requested."],
        "keywords": ["ID card", "Aadhaar card", "photocopy"],
        "action_items": [{"owner": "Speaker_B", "action": "Bring a photocopy of the Aadhaar card.", "evidence_line_ids": [2]}],
    }
    return json.dumps(chunk, ensure_ascii=False), json.dumps(synthesis, ensure_ascii=False)


def test_parse_json_object_accepts_code_fence():
    assert parse_json_object("```json\n{\"x\": 1}\n```") == {"x": 1}


def test_guardrail_restores_immutable_metadata_and_drops_hallucinated_ids():
    chunk, synthesis = model_outputs()
    report = PostProcessor(StaticLLM([chunk, synthesis])).process_transcript(transcript())
    assert [x.line_id for x in report.cleaned_dialogue] == [1, 2]
    assert report.cleaned_dialogue[0].speaker == "Speaker_A"
    assert report.cleaned_dialogue[0].start == 0.0
    assert report.cleaned_dialogue[0].end == 2.0
    assert report.cleaned_dialogue[0].text == "आई डी कार्ड गुम हो गया है।"
    assert report.action_items[0].evidence_line_ids == [2]
    assert report.uncertain_lines == []


def test_missing_line_is_restored_and_flagged():
    chunk, synthesis = model_outputs()
    obj = json.loads(chunk, object_pairs_hook=dict)
    obj["cleaned_dialogue"] = obj["cleaned_dialogue"][:1]
    report = PostProcessor(StaticLLM([json.dumps(obj, ensure_ascii=False), synthesis])).process_transcript(transcript())
    assert [x.line_id for x in report.cleaned_dialogue] == [1, 2]
    assert 2 in report.uncertain_lines
    assert "omitted" in report.cleaned_dialogue[1].note.lower()

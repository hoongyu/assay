from assay.checks.runner import BATCH_CHECKS, EPISODE_CHECKS
from assay.diagnose import LABEL_EN, Label
from assay.i18n import CHECKS, check_name
from assay.types import CAUSE_EN, SOURCE_EN, SOURCE_NAMES, Cause, source_name


def test_every_cause_source_and_label_has_english():
    assert set(CAUSE_EN) == set(Cause)
    assert set(LABEL_EN) == set(Label)
    assert set(SOURCE_EN) == set(SOURCE_NAMES)
    assert source_name("meta") == "元数据" and source_name("meta", "en") == "metadata"


def test_check_names_cover_every_rule():
    rules = set(EPISODE_CHECKS) | set(BATCH_CHECKS) | {"wrong_task", "progress_drop", "video_judges"}
    assert rules <= set(CHECKS)


def test_check_name_in_both_languages():
    assert check_name("regrasp") == "二次抓取" and check_name("regrasp", "en") == "regrasp"
    assert check_name("topreward.progress_drop") == "TOPReward的「进度回落」"
    assert check_name("topreward.progress_drop", "en") == "TOPReward progress drop"
    assert check_name("video_judges", "en") == "video judges"

"""Finding the KSL Dictionary scrape's saved videos (dictionary.py)."""

from ashiria_train.dictionary import folder_index, is_video, resolve_copy

MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 6000
ERROR_PAGE = b"<!DOCTYPE HTML>\n<html>" + b" " * 6000


def test_folder_matched_by_number_when_the_name_was_changed(tmp_path):
    # The scraper wrote "210_Are You Afraid?"; the copy on disk has "_" for "?".
    (tmp_path / "210_Are You Afraid_").mkdir()
    (tmp_path / "210_Are You Afraid_" / "3026.mp4").write_bytes(MP4)
    (tmp_path / "1_A").mkdir()
    (tmp_path / "1_A" / "1510.mp4").write_bytes(MP4)
    folders = folder_index(tmp_path)
    assert resolve_copy(tmp_path, folders, "ksl_videos/210_Are You Afraid?/3026.mp4", 210) == "210_Are You Afraid_/3026.mp4"
    assert resolve_copy(tmp_path, folders, "ksl_videos/1_A/1510.mp4", "1") == "1_A/1510.mp4"
    assert resolve_copy(tmp_path, folders, "ksl_videos/2_B/9.mp4", 2) is None
    assert resolve_copy(tmp_path, folders, "", 3) is None


def test_error_pages_and_truncated_files_are_not_videos(tmp_path):
    for name, data in (("ok.mp4", MP4), ("404.mp4", ERROR_PAGE), ("cut.mp4", MP4[:900])):
        (tmp_path / name).write_bytes(data)
    assert is_video(tmp_path / "ok.mp4")
    assert not is_video(tmp_path / "404.mp4")
    assert not is_video(tmp_path / "cut.mp4")

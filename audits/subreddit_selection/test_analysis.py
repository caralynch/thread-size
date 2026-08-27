from __future__ import annotations

import unittest

from analyze_selection_sources import account_summary, classify_format, is_root


class AuditLogicTests(unittest.TestCase):
    def test_root_requires_parent_and_identity_agreement(self) -> None:
        self.assertTrue(is_root({"parent": "", "id": "a", "thread_id": "a"}))
        self.assertFalse(is_root({"parent": "x", "id": "a", "thread_id": "a"}))
        self.assertFalse(is_root({"parent": "", "id": "a", "thread_id": "b"}))

    def test_external_link_is_not_equated_with_media(self) -> None:
        category, reason = classify_format(
            {"domain": "example.org", "url": "https://example.org/article", "body": ""},
            "url_domain_only",
        )
        self.assertEqual(category, "external_article_or_link")
        self.assertEqual(reason, "external_nonmedia_host")

    def test_non_reddit_media_hosts_are_detected(self) -> None:
        image, _ = classify_format(
            {"domain": "imgur.com", "url": "https://imgur.com/example.jpg"},
            "url_domain_only",
        )
        video, _ = classify_format(
            {"domain": "youtube.com", "url": "https://youtube.com/watch?v=example"},
            "url_domain_only",
        )
        self.assertEqual(image, "image")
        self.assertEqual(video, "video")

    def test_repeat_account_denominator_and_numerator(self) -> None:
        denominator, numerator, percent = account_summary({"a": 1, "b": 2, "c": 4})
        self.assertEqual(denominator, 3)
        self.assertEqual(numerator, 2)
        self.assertAlmostEqual(percent, 200 / 3)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_TEMPLATE = BASE_DIR / "templates" / "index.html"
ADMIN_TEMPLATE = BASE_DIR / "templates" / "admin.html"


class OfflineTemplateTests(unittest.TestCase):
    def test_player_template_uses_socketio_cdn_with_local_fallback(self):
        template = INDEX_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js", template)
        self.assertIn("socketio-lite.js", template)
        self.assertNotIn('/socket.io/socket.io.js', template)
        # Запасной клиент — через document.write при разборе страницы. Смена src
        # в onerror не работает: уже запущенный <script> повторно не грузится.
        self.assertIn("window.io || document.write", template)
        self.assertNotIn("this.src=", template)

    def test_admin_template_uses_socketio_cdn_with_local_fallback(self):
        template = ADMIN_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js", template)
        self.assertIn("socketio-lite.js", template)
        self.assertNotIn('/socket.io/socket.io.js', template)
        # Запасной клиент — через document.write при разборе страницы. Смена src
        # в onerror не работает: уже запущенный <script> повторно не грузится.
        self.assertIn("window.io || document.write", template)
        self.assertNotIn("this.src=", template)

    def test_player_names_are_escaped_before_innerhtml(self):
        for template_path in (INDEX_TEMPLATE, ADMIN_TEMPLATE):
            template = template_path.read_text(encoding="utf-8")
            with self.subTest(template=template_path.name):
                self.assertIn("function escapeHtml(", template)
                self.assertNotIn("${p.name}", template)
                self.assertNotIn("${g.game_id}", template)

    def test_templates_include_local_offline_fallback_stylesheet(self):
        expected_link = "{{ url_for('static', filename='offline-fallback.css') }}"

        self.assertIn(expected_link, INDEX_TEMPLATE.read_text(encoding="utf-8"))
        self.assertIn(expected_link, ADMIN_TEMPLATE.read_text(encoding="utf-8"))

    def test_templates_restore_primary_visual_cdns(self):
        required_markers = (
            "fonts.googleapis.com/css2?family=Outfit",
            "cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
            "cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        )

        for template_path in (INDEX_TEMPLATE, ADMIN_TEMPLATE):
            template = template_path.read_text(encoding="utf-8")
            for marker in required_markers:
                with self.subTest(template=template_path.name, marker=marker):
                    self.assertIn(marker, template)

    def test_admin_template_does_not_require_bootstrap_bundle(self):
        template = ADMIN_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js", template)
        self.assertIn("window.bootstrap = window.bootstrap || {}", template)

    def test_local_socketio_lite_client_exists(self):
        client_path = BASE_DIR / "static" / "socketio-lite.js"

        self.assertTrue(client_path.exists())
        self.assertIn("class LiteSocket", client_path.read_text(encoding="utf-8"))

    def test_player_question_text_supports_line_breaks(self):
        template = INDEX_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("white-space: pre-line;", template)

    def test_player_template_contains_answer_validation_ui(self):
        template = INDEX_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('id="answerError"', template)
        self.assertIn("answer_validation_error", template)
        self.assertIn("MIN 4", template.upper())

    def test_player_template_contains_track_meta_placeholder(self):
        template = INDEX_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('id="questionTrackMetaDisplay"', template)
        self.assertIn('id="resultTrackMetaDisplay"', template)
        self.assertIn("renderTrackMeta(", template)


class HostAndPlayerSessionTests(unittest.TestCase):
    def test_admin_password_is_sent_on_connect(self):
        # Пароль уходит в auth при подключении: клики, накопленные за разрыв,
        # приходят раньше join_admin и без этого отклонялись бы.
        admin = ADMIN_TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("io({auth: (cb) => cb({admin: true, key: getAdminKey()})})", admin)
        lite = (BASE_DIR / "static" / "socketio-lite.js").read_text(encoding="utf-8")
        self.assertIn("_sendConnect()", lite)
        self.assertNotIn('this._sendRaw("40");', lite)

    def test_player_sees_join_errors_and_server_name(self):
        template = INDEX_TEMPLATE.read_text(encoding="utf-8")
        # Имя обрезается сервером до 32 символов — поле не даёт ввести больше.
        self.assertIn('id="usernameInput" class="form-control form-control-custom mb-3" '
                      'placeholder="Your Name" maxlength="32"', template)
        # Отказ на автоматический вход возвращает экран входа, где видна причина.
        self.assertIn("showScreen('loginBlock');", template)
        self.assertIn("myName = d.name;", template)
        # VIP приходит вместе с разрешением листать: прежний мог уйти.
        self.assertIn("if (d && d.vip_id) amIVip = (d.vip_id === socket.id);", template)


if __name__ == "__main__":
    unittest.main()

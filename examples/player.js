/* KokoroTTS-HF Studio wall — original code. No upstream parts.
   Renders language filters + 70 voice cards from window.VOICE_EXAMPLES.
   Owns the VI/EN language toggle for the whole page. */
(function () {
  "use strict";

  var LANG_ORDER = [
    "American English", "British English", "Japanese", "Mandarin Chinese",
    "Spanish", "French", "Hindi", "Italian", "Brazilian Portuguese",
    "German", "Vietnamese"
  ];
  var LANG_VI = {
    "American English": "Tiếng Anh (Mỹ)", "British English": "Tiếng Anh (Anh)",
    "Japanese": "Tiếng Nhật", "Mandarin Chinese": "Tiếng Trung",
    "Spanish": "Tiếng Tây Ban Nha", "French": "Tiếng Pháp", "Hindi": "Tiếng Hindi",
    "Italian": "Tiếng Ý", "Brazilian Portuguese": "Tiếng Bồ Đào Nha (Brasil)",
    "German": "Tiếng Đức", "Vietnamese": "Tiếng Việt"
  };
  var LIVE_RE = /^(af_|am_|bf_|bm_)/;

  var STR = {
    vi: {
      tagline: "70 giọng đọc · 11 ngôn ngữ · CPU",
      eyebrow: "Studio TTS · chạy hoàn toàn trên CPU",
      hero: "Nói bằng trình duyệt của bạn",
      heroSub: "Gõ chữ, chọn 1 trong 70 giọng, bấm Tạo tiếng — 28 giọng Anh tổng hợp trực tiếp trên máy bạn (CPU), các giọng còn lại phát mẫu studio tức thì. Không server, không quota.",
      heroSubEn: "Type text, pick any of 70 voices, press Generate — 28 English voices synthesize live on your CPU, the rest play studio samples instantly. No server, no quota.",
      pillVoices: "giọng · 11 ngôn ngữ",
      studioTitle: "Studio — tạo tiếng nói",
      studioSub: "28 giọng tiếng Anh chạy live 100% trên CPU máy bạn. 42 giọng còn lại tổng hợp qua backend CPU — trang tự tìm backend local, chưa có thì làm theo hướng dẫn bên dưới.",
      textLabel: "Văn bản", modeLabel: "Chế độ", mSingle: "1 giọng", mMix: "Pha trộn A + B",
      all70: "cả 70 giọng", mixLabel: "Tỉ lệ giọng B",
      mixNote: "Mix dựng A và B riêng rồi hòa âm thanh. Qua backend thì blend đúng embedding.",
      spdLabel: "Tốc độ", presetLabel: "Câu mẫu nhanh",
      beTitle: "Backend CPU tự host (tùy chọn — cho 42 giọng phi-Anh live)",
      beNote: "Để trống rồi bấm Lưu = tự tìm backend local. Hoặc dán địa chỉ backend của bạn. Muốn ngắt hẳn thì bấm Ngắt kết nối.",
      beSave: "Lưu", beOffBtn: "Ngắt kết nối", beCopyBtn: "Chép lệnh docker", beEmpty: "Đã xóa backend — dùng mẫu có sẵn.", wallTitle: "Chọn giọng — 70 voices",
      wallSub: "Bấm ▶ nghe mẫu, bấm “Dùng giọng này” để đưa lên Studio. Thẻ xanh lá = tổng hợp live trên CPU.",
      all: "Tất cả", use: "Dùng giọng này ↑", live: "live CPU", sample: "mẫu",
      play: "Nghe mẫu", pause: "Dừng"
    },
    en: {
      tagline: "70 voices · 11 languages · CPU",
      eyebrow: "TTS studio · runs fully on CPU",
      hero: "Speak in your browser",
      heroSub: "Type text, pick any of 70 voices, press Generate — 28 English voices synthesize live on your CPU, the rest play studio samples instantly. No server, no quota.",
      heroSubEn: "Gõ chữ, chọn 1 trong 70 giọng, bấm Tạo tiếng — 28 giọng Anh live trên CPU, các giọng còn lại phát mẫu tức thì. Không server, không quota.",
      pillVoices: "voices · 11 languages",
      studioTitle: "Studio — generate speech",
      studioSub: "28 English voices run live, 100% on your CPU. The other 42 synthesize via the CPU backend — the page auto-detects a local backend, otherwise follow the guide below.",
      textLabel: "Text", modeLabel: "Mode", mSingle: "Single voice", mMix: "Blend A + B",
      all70: "all 70 voices", mixLabel: "Voice B weight",
      mixNote: "Mix renders A and B separately, then blends the audio. Via backend it blends true embeddings.",
      spdLabel: "Speed", presetLabel: "Quick text",
      beTitle: "Self-hosted CPU backend (optional — live non-English voices)",
      beNote: "Empty + Save = auto-detect a local backend. Or paste your backend address. Disconnect cuts it off entirely.",
      beSave: "Save", beOffBtn: "Disconnect", beCopyBtn: "Copy docker command", beEmpty: "Backend cleared — using built-in samples.", wallTitle: "Pick a voice — 70 voices",
      wallSub: "Press ▶ to hear the sample, “Use this voice” loads it into the Studio. Green tag = live synthesis on CPU.",
      all: "All", use: "Use this voice ↑", live: "live CPU", sample: "sample",
      play: "Preview", pause: "Pause"
    }
  };

  window.KokoroLang = localStorage.getItem("kokorotts_hf_lang") || "vi";
  function t(key) { return (STR[window.KokoroLang] || STR.vi)[key] || key; }
  window.KokoroTr = t;

  function applyLang() {
    document.documentElement.lang = window.KokoroLang;
    var els = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < els.length; i++) {
      var k = els[i].getAttribute("data-i18n");
      if (STR[window.KokoroLang][k]) els[i].textContent = STR[window.KokoroLang][k];
    }
    var tg = document.getElementById("langToggle");
    if (tg) tg.textContent = window.KokoroLang === "vi" ? "EN" : "VI";
    if (typeof window.KokoroRefreshTexts === "function") window.KokoroRefreshTexts();
  }

  var activeLang = "American English";

  function voices() {
    return Array.isArray(window.VOICE_EXAMPLES) ? window.VOICE_EXAMPLES : [];
  }
  function langName(en) {
    return window.KokoroLang === "vi" ? (LANG_VI[en] || en) : en;
  }
  function isLive(id) { return LIVE_RE.test(id || ""); }

  function renderFilters() {
    var box = document.getElementById("filters");
    if (!box) return;
    var counts = {};
    voices().forEach(function (v) { counts[v.language] = (counts[v.language] || 0) + 1; });
    var html = '<button class="btn" type="button" data-f="all">' + t("all") + " (" + voices().length + ")</button>";
    LANG_ORDER.forEach(function (l) {
      if (!counts[l]) return;
      html += '<button class="btn" type="button" data-f="' + l + '">' + langName(l) + " (" + counts[l] + ")</button>";
    });
    box.innerHTML = html;
    var btns = box.querySelectorAll("button");
    btns.forEach(function (b) {
      b.classList.toggle("on", b.getAttribute("data-f") === activeLang || (activeLang === "all" && b.getAttribute("data-f") === "all"));
      b.addEventListener("click", function () {
        activeLang = b.getAttribute("data-f");
        btns.forEach(function (x) { x.classList.toggle("on", x === b); });
        renderWall();
      });
    });
  }

  function stopOthers(current) {
    document.querySelectorAll("#wall audio").forEach(function (a) {
      if (a !== current) { try { a.pause(); } catch (e) {} }
    });
  }

  function renderWall() {
    var wall = document.getElementById("wall");
    if (!wall) return;
    var list = voices().filter(function (v) {
      return activeLang === "all" || v.language === activeLang;
    });
    var html = "";
    list.forEach(function (v, i) {
      var live = isLive(v.voice);
      html += '<article class="vcard">' +
        '<div class="nm">' + escapeHtml(v.name) + "</div>" +
        '<div class="lg">' + escapeHtml(langName(v.language)) + "</div>" +
        '<div class="vmeta"><span class="id">' + escapeHtml(v.voice) + "</span>" +
        '<span class="tag ' + (live ? "live" : "sample") + '">' + (live ? t("live") : t("sample")) + "</span></div>" +
        '<audio preload="none" src="' + escapeAttr(v.file) + '"></audio>' +
        '<div style="display:flex;gap:8px;margin-top:10px">' +
        '<button class="btn use" type="button" data-play="' + i + '" style="flex:1">▶ ' + t("play") + "</button>" +
        '<button class="btn use" type="button" data-use="' + escapeAttr(v.voice) + '" style="flex:2">' + t("use") + "</button>" +
        "</div></article>";
    });
    wall.innerHTML = html || "<p>—</p>";

    wall.querySelectorAll("[data-play]").forEach(function (b) {
      b.addEventListener("click", function () {
        var card = b.closest(".vcard");
        var a = card ? card.querySelector("audio") : null;
        if (!a) return;
        if (a.paused) { stopOthers(a); a.play().catch(function () {}); b.innerHTML = "⏸ " + t("pause"); }
        else { a.pause(); b.innerHTML = "▶ " + t("play"); }
        a.onended = function () { b.innerHTML = "▶ " + t("play"); };
        a.onpause = function () { b.innerHTML = "▶ " + t("play"); };
      });
    });
    wall.querySelectorAll("[data-use]").forEach(function (b) {
      b.addEventListener("click", function () {
        if (typeof window.KokoroUseVoice === "function") window.KokoroUseVoice(b.getAttribute("data-use"));
      });
    });
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function escapeAttr(s) { return escapeHtml(s).replace(/'/g, "&#39;"); }

  function refreshWall() { renderFilters(); renderWall(); }
  window.KokoroRefreshWall = refreshWall;

  document.addEventListener("DOMContentLoaded", function () {
    var tg = document.getElementById("langToggle");
    if (tg) tg.addEventListener("click", function () {
      window.KokoroLang = window.KokoroLang === "vi" ? "en" : "vi";
      try { localStorage.setItem("kokorotts_hf_lang", window.KokoroLang); } catch (e) {}
      applyLang(); refreshWall();
    });
    applyLang();
    refreshWall();
  });
})();

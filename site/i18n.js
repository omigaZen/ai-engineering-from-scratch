/**
 * Small client-side i18n helper for the static curriculum site.
 *
 * English remains canonical. Chinese strings are optional overlays from
 * data.js and lesson docs; missing translations fall back to English.
 */
(function () {
  'use strict';

  var SUPPORTED = { en: true, zh: true };
  var STORAGE_KEY = 'aifs_lang';

  var TEXT = {
    en: {
      'a11y.skip': 'Skip to content',
      'nav.contents': 'Contents',
      'nav.catalog': 'Catalog',
      'nav.roadmap': 'Roadmap',
      'nav.glossary': 'Glossary',
      'nav.about': 'About',
      'nav.home': 'Home',
      'nav.github': 'GitHub',
      'nav.report': 'Report / Suggest',
      'nav.report.short': 'Report',
      'search.label': 'Search (⌘K)',
      'theme.toggle': 'Toggle theme',
      'lang.toggle': '中文',
      'lang.toggleLabel': 'Switch language',
      'phase': 'Phase',
      'status.complete': 'Complete',
      'status.in-progress': 'In Progress',
      'status.planned': 'Planned',
      'status.completed': 'completed',
      'action.read': 'Read',
      'action.review': 'Review',
      'action.install': 'Install',
      'action.copy': 'Copy',
      'action.copied': 'Copied!',
      'action.copyCommand': 'Copy command',
      'action.viewGithub': 'View on GitHub',
      'action.viewLessonGithub': 'View lesson on GitHub',
      'footer.site': 'AI Engineering from Scratch · open source · free forever.',
      'footer.short': '© 2026 · open source · free forever',
      'home.meta.left': 'FIG_000 · curriculum v1.0 · 2026',
      'home.meta.right': 'open source · MIT',
      'home.tagline': '503 lessons. 20 phases. Every algorithm built from raw math before a single framework gets imported.',
      'home.attribution': 'Maintained by Rohit Ghumare and contributors. Run on your own machine.',
      'home.star': 'Star on GitHub',
      'home.follow': 'Follow @rohitg00',
      'home.preface.title': 'How this works',
      'home.preface.p1': "Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can't explain its loss curve. You hook a function to an agent but can't say what attention does inside the model that's calling it.",
      'home.preface.p2': 'This curriculum is the spine. 20 phases, 503 lessons, four languages: Python, TypeScript, Rust, Julia. Linear algebra at one end, autonomous swarms at the other. Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it is doing under the hood.',
      'home.preface.p3': 'Each lesson runs the same loop: read the problem, derive the math, write the code, run the test, keep the artifact. No five-minute videos, no copy-paste deploys, no hand-holding. Free, open source, and built to run on your own laptop.',
      'home.progress.title': 'Current Progress',
      'home.progress.finished': 'Finished Lessons',
      'home.progress.phases': 'Phases',
      'home.progress.languages': 'Languages',
      'home.progress.glossary': 'Glossary Terms',
      'home.contents.title': 'Curriculum · 20 phases · 503 lessons',
      'home.contents.subtitle': 'Tap a phase to expand its lessons. Each one ships when its math, code, and test are all written.',
      'home.modal.saved': 'Progress saved in browser only',
      'home.modal.reset': 'Reset progress',
      'home.modal.confirmReset': 'Clear all your local progress (quiz answers and completed lessons)? This cannot be undone.',
      'home.colophon.title': 'Colophon',
      'home.colophon.body': 'The entire curriculum is on GitHub. Clone it, fork it, learn at your own pace. No paywall, no signup. Every lesson has runnable code in Python, TypeScript, Rust, or Julia, depending on what fits the concept best.',
      'catalog.title': 'Lesson Catalog',
      'catalog.subtitle': 'Every lesson across all 20 phases. Search, filter, sort.',
      'catalog.search': 'Search lessons...',
      'catalog.allPhases': 'All Phases',
      'catalog.allStatus': 'All Status',
      'catalog.phase': 'Phase',
      'catalog.lesson': 'Lesson',
      'catalog.type': 'Type',
      'catalog.language': 'Language',
      'catalog.status': 'Status',
      'catalog.count': '{shown} of {total} lessons',
      'catalog.empty': 'No lessons match your filters.',
      'roadmap.title': 'Roadmap',
      'roadmap.subtitle': 'Click any phase to see its prerequisites and what it unlocks downstream.',
      'roadmap.clear': '✕ Clear selection',
      'roadmap.scroll': '↔ Scroll to explore the full graph',
      'roadmap.none': 'None. This is a starting point.',
      'roadmap.final': 'Final destination. End of the curriculum.',
      'roadmap.prerequisites': 'Prerequisites',
      'roadmap.unlocks': 'Unlocks',
      'roadmap.lessonsComplete': '{done} of {total} lessons complete',
      'roadmap.prereqPhases': '{count} prerequisite phases',
      'roadmap.unlockedPhases': '{count} phases unlocked',
      'glossary.title': 'AI Glossary',
      'glossary.subtitle': 'What people say vs what things actually mean',
      'glossary.search': 'Search terms...',
      'glossary.count': '{shown} of {total} terms',
      'glossary.empty': 'No terms match your search.',
      'glossary.says': 'What people say',
      'glossary.means': 'What it actually means',
      'lesson.loading': 'Loading lesson...',
      'lesson.noPathTitle': 'No lesson path specified',
      'lesson.noPathBody': 'Add ?path=phases/01-math-foundations/01-linear-algebra-intuition to the URL.',
      'lesson.renderErrorTitle': 'Render error',
      'lesson.renderErrorBody': 'Loaded the lesson markdown but failed to render it. Details in the browser console.',
      'lesson.notFoundTitle': 'Lesson not found',
      'lesson.notFoundBody': 'Could not fetch the lesson at <code>{path}</code>. It may not have been written yet.',
      'lesson.translationNotFoundTitle': 'Chinese translation not found',
      'lesson.translationNotFoundBody': 'Could not fetch the Chinese lesson document at <code>{path}</code>. The page will not silently fall back to English; publish the translated Markdown to main or switch to English.',
      'lesson.backHome': 'Back to Home',
      'lesson.toc': 'On this page',
      'lesson.learningObjectives': 'Learning Objectives',
      'lesson.labChallenge': 'Lab Challenge',
      'lesson.copy': 'Copy',
      'lesson.copied': 'Copied!',
      'lesson.diagram': 'Diagram',
      'lesson.expand': 'Expand',
      'lesson.diagramError': 'Diagram could not be rendered.',
      'quiz.pre': 'Pre-Lesson Check',
      'quiz.check': 'Mid-Lesson Check',
      'quiz.post': 'Post-Lesson Quiz',
      'quiz.all': 'Quiz',
      'quiz.correct': '{correct}/{total} correct',
      'lesson.prev': '← Previous',
      'lesson.next': 'Next →',
      'panel.outputs.title': 'What This Lesson Ships',
      'panel.outputs.subtitle': 'Prompts, skills, and artifacts you can use right now',
      'panel.outputs.loading': 'Loading outputs...',
      'panel.outputs.prompt': 'Prompt',
      'panel.outputs.skill': 'Skill',
      'panel.outputs.output': 'Output',
      'panel.outputs.loadingDesc': 'Loading description...',
      'panel.outputs.promptHint': 'Paste into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads prompts',
      'panel.code.title': 'Run the Code',
      'panel.code.subtitle': 'Executable files from this lesson',
      'panel.code.loading': 'Loading code files...',
      'panel.quiz.title': 'Test Your Understanding',
      'panel.quiz.subtitle': 'Did you get it?',
      'panel.quiz.question': 'Question {current} of {total}',
      'panel.quiz.complete': 'Complete all questions to see your score',
      'panel.quiz.perfect': 'Perfect score!',
      'panel.quiz.good': 'Great work!',
      'panel.quiz.study': 'Keep studying!',
      'panel.quiz.deeper': 'Want a deeper quiz? Run <code>/check-understanding {phase}</code> in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed',
      'panel.path.title': 'Learning Path',
      'panel.path.earlier': '{count} earlier lessons',
      'panel.path.later': '{count} later lessons',
      'panel.path.progress': "You've completed {done} of {total} lessons in this phase",
      'panel.path.ready': 'Ready for Phase {phase}: {name}',
      'panel.continue.title': 'Continue Learning',
      'panel.continue.finished': 'You finished this phase!',
      'panel.continue.browse': 'Browse all Phase {phase} lessons',
      'panel.continue.catalog': 'Full course catalog',
      'panel.continue.callout': 'Run <code>/find-your-level</code> in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed for a personalized learning path',
      'palette.label': 'Search lessons and glossary',
      'palette.input': 'Search lessons and glossary...',
      'palette.empty': 'Type to search 503 lessons, 499 outputs, and glossary terms',
      'palette.noResults': 'No results for <em>{query}</em>',
      'palette.navigate': 'navigate',
      'palette.open': 'open',
      'palette.close': 'close',
      'palette.glossary': 'Glossary',
      'palette.artifact': 'Artifact'
    },
    zh: {
      'a11y.skip': '跳到正文',
      'nav.contents': '目录',
      'nav.catalog': '课程表',
      'nav.roadmap': '路线图',
      'nav.glossary': '术语表',
      'nav.about': '关于',
      'nav.home': '首页',
      'nav.github': 'GitHub',
      'nav.report': '反馈 / 建议',
      'nav.report.short': '反馈',
      'search.label': '搜索 (⌘K)',
      'theme.toggle': '切换主题',
      'lang.toggle': 'EN',
      'lang.toggleLabel': '切换语言',
      'phase': '阶段',
      'status.complete': '已完成',
      'status.in-progress': '进行中',
      'status.planned': '计划中',
      'status.completed': '已完成',
      'action.read': '阅读',
      'action.review': '复习',
      'action.install': '安装',
      'action.copy': '复制',
      'action.copied': '已复制',
      'action.copyCommand': '复制命令',
      'action.viewGithub': '在 GitHub 查看',
      'action.viewLessonGithub': '在 GitHub 查看课程',
      'footer.site': 'AI Engineering from Scratch · 开源 · 永久免费。',
      'footer.short': '© 2026 · 开源 · 永久免费',
      'home.meta.left': 'FIG_000 · 课程 v1.0 · 2026',
      'home.meta.right': '开源 · MIT',
      'home.tagline': '503 节课，20 个阶段。先从原始数学手写每个算法，再引入框架。',
      'home.attribution': '由 Rohit Ghumare 和贡献者维护。可在你自己的机器上运行。',
      'home.star': '在 GitHub 点星',
      'home.follow': '关注 @rohitg00',
      'home.preface.title': '课程如何运作',
      'home.preface.p1': '大多数 AI 资料都是碎片化的：一篇论文、一篇 fine-tuning 博客、一个炫目的 agent demo。它们很少连成体系。你可能能上线 chatbot，却说不清 loss curve；你能给 agent 接 function，却解释不了模型内部的 attention。',
      'home.preface.p2': '这套课程提供主干。20 个阶段，503 节课，覆盖 Python、TypeScript、Rust、Julia。起点是 linear algebra，终点是 autonomous swarms。每个算法都先从原始数学手写：backprop、tokenizer、attention、agent loop。等 PyTorch 出现时，你已经知道它在底层做什么。',
      'home.preface.p3': '每节课都走同一个循环：读问题，推数学，写代码，跑测试，保留可复用产物。没有五分钟视频，没有复制粘贴部署，没有过度手把手。免费、开源，并且设计为能在你的 laptop 上运行。',
      'home.progress.title': '当前进度',
      'home.progress.finished': '已完成课程',
      'home.progress.phases': '阶段',
      'home.progress.languages': '语言',
      'home.progress.glossary': '术语数量',
      'home.contents.title': '课程 · 20 个阶段 · 503 节课',
      'home.contents.subtitle': '点击阶段展开课程。每节课在数学、代码和测试都完成后交付。',
      'home.modal.saved': '进度只保存在当前浏览器',
      'home.modal.reset': '重置进度',
      'home.modal.confirmReset': '清除本地进度（quiz 答案和已完成课程）？此操作无法撤销。',
      'home.colophon.title': '说明',
      'home.colophon.body': '整套课程都在 GitHub 上。clone、fork，按自己的节奏学习。没有付费墙，没有注册。每节课都有可运行代码，语言会按概念需要选用 Python、TypeScript、Rust 或 Julia。',
      'catalog.title': '课程表',
      'catalog.subtitle': '跨 20 个阶段查看、搜索、筛选和排序所有课程。',
      'catalog.search': '搜索课程...',
      'catalog.allPhases': '全部阶段',
      'catalog.allStatus': '全部状态',
      'catalog.phase': '阶段',
      'catalog.lesson': '课程',
      'catalog.type': '类型',
      'catalog.language': '语言',
      'catalog.status': '状态',
      'catalog.count': '{shown} / {total} 节课',
      'catalog.empty': '没有课程匹配当前筛选。',
      'roadmap.title': '路线图',
      'roadmap.subtitle': '点击任意阶段，查看它的前置依赖以及后续解锁内容。',
      'roadmap.clear': '✕ 清除选择',
      'roadmap.scroll': '↔ 横向滚动查看完整图',
      'roadmap.none': '无。这是起点。',
      'roadmap.final': '终点。课程到这里结束。',
      'roadmap.prerequisites': '前置阶段',
      'roadmap.unlocks': '解锁内容',
      'roadmap.lessonsComplete': '{done} / {total} 节课已完成',
      'roadmap.prereqPhases': '{count} 个前置阶段',
      'roadmap.unlockedPhases': '解锁 {count} 个阶段',
      'glossary.title': 'AI 术语表',
      'glossary.subtitle': '大家常说的话 vs 实际含义',
      'glossary.search': '搜索术语...',
      'glossary.count': '{shown} / {total} 个术语',
      'glossary.empty': '没有术语匹配当前搜索。',
      'glossary.says': '大家常说',
      'glossary.means': '实际含义',
      'lesson.loading': '正在加载课程...',
      'lesson.noPathTitle': '缺少课程路径',
      'lesson.noPathBody': '请在 URL 中添加 ?path=phases/01-math-foundations/01-linear-algebra-intuition。',
      'lesson.renderErrorTitle': '渲染错误',
      'lesson.renderErrorBody': '课程 Markdown 已加载，但渲染失败。详细信息见浏览器 console。',
      'lesson.notFoundTitle': '找不到课程',
      'lesson.notFoundBody': '无法获取 <code>{path}</code> 处的课程。它可能还没有写完。',
      'lesson.translationNotFoundTitle': '中文翻译未找到',
      'lesson.translationNotFoundBody': '无法获取中文课程文档 <code>{path}</code>。页面不会静默回退英文；请将翻译 Markdown 发布到 main，或切换到英文。',
      'lesson.backHome': '返回首页',
      'lesson.toc': '本页目录',
      'lesson.learningObjectives': '学习目标',
      'lesson.labChallenge': '实验挑战',
      'lesson.copy': '复制',
      'lesson.copied': '已复制',
      'lesson.diagram': '图示',
      'lesson.expand': '展开',
      'lesson.diagramError': '图示无法渲染。',
      'quiz.pre': '课前检查',
      'quiz.check': '课中检查',
      'quiz.post': '课后测验',
      'quiz.all': '测验',
      'quiz.correct': '{correct}/{total} 正确',
      'lesson.prev': '← 上一课',
      'lesson.next': '下一课 →',
      'panel.outputs.title': '本课交付什么',
      'panel.outputs.subtitle': '可以立即复用的 prompts、skills 和 artifacts',
      'panel.outputs.loading': '正在加载输出...',
      'panel.outputs.prompt': 'Prompt',
      'panel.outputs.skill': 'Skill',
      'panel.outputs.output': 'Output',
      'panel.outputs.loadingDesc': '正在加载描述...',
      'panel.outputs.promptHint': '粘贴到 Claude、Cursor、Codex、OpenClaw、Hermes，或任何读取 prompts 的 agent',
      'panel.code.title': '运行代码',
      'panel.code.subtitle': '本课提供的可执行文件',
      'panel.code.loading': '正在加载代码文件...',
      'panel.quiz.title': '检查理解',
      'panel.quiz.subtitle': '你真的掌握了吗？',
      'panel.quiz.question': '第 {current} / {total} 题',
      'panel.quiz.complete': '完成所有题目后查看得分',
      'panel.quiz.perfect': '满分！',
      'panel.quiz.good': '做得不错！',
      'panel.quiz.study': '继续复习。',
      'panel.quiz.deeper': '想做更深入的 quiz？在 Claude、Cursor、Codex、OpenClaw、Hermes 或任何安装课程 skills 的 agent 中运行 <code>/check-understanding {phase}</code>',
      'panel.path.title': '学习路径',
      'panel.path.earlier': '前面 {count} 节课',
      'panel.path.later': '后面 {count} 节课',
      'panel.path.progress': '你已完成本阶段 {done} / {total} 节课',
      'panel.path.ready': '准备进入阶段 {phase}: {name}',
      'panel.continue.title': '继续学习',
      'panel.continue.finished': '你已完成这个阶段！',
      'panel.continue.browse': '浏览阶段 {phase} 的全部课程',
      'panel.continue.catalog': '完整课程表',
      'panel.continue.callout': '在 Claude、Cursor、Codex、OpenClaw、Hermes 或任何安装课程 skills 的 agent 中运行 <code>/find-your-level</code>，生成个性化学习路径',
      'palette.label': '搜索课程和术语表',
      'palette.input': '搜索课程和术语表...',
      'palette.empty': '输入关键词，搜索 503 节课、499 个输出和术语表',
      'palette.noResults': '没有找到 <em>{query}</em>',
      'palette.navigate': '导航',
      'palette.open': '打开',
      'palette.close': '关闭',
      'palette.glossary': '术语',
      'palette.artifact': '产物'
    }
  };

  function normalizeLang(lang) {
    lang = String(lang || '').toLowerCase();
    if (lang === 'zh-cn' || lang === 'zh_hans' || lang === 'zh-hans') return 'zh';
    return SUPPORTED[lang] ? lang : '';
  }

  function queryLang() {
    try {
      return normalizeLang(new URLSearchParams(window.location.search).get('lang'));
    } catch (_) {
      return '';
    }
  }

  function getLang() {
    var fromQuery = queryLang();
    if (fromQuery) {
      try { localStorage.setItem(STORAGE_KEY, fromQuery); } catch (_) {}
      return fromQuery;
    }
    try {
      return normalizeLang(localStorage.getItem(STORAGE_KEY)) || 'en';
    } catch (_) {
      return 'en';
    }
  }

  function t(key, vars) {
    var lang = getLang();
    var msg = (TEXT[lang] && TEXT[lang][key]) || TEXT.en[key] || key;
    if (vars) {
      msg = msg.replace(/\{([a-zA-Z0-9_]+)\}/g, function (_, name) {
        return vars[name] == null ? '' : String(vars[name]);
      });
    }
    return msg;
  }

  function withLangHref(url) {
    if (!url || /^(https?:|mailto:|data:|javascript:)/i.test(url)) return url;
    if (url.charAt(0) === '#') return url;
    var lang = getLang();
    try {
      var u = new URL(url, window.location.href);
      u.searchParams.set('lang', lang);
      var base = u.pathname.split('/').pop() || u.pathname;
      var rel = base + u.search + u.hash;
      if (url.indexOf('/') >= 0 && !url.match(/^[^/?#]+\.html/)) {
        rel = u.pathname.replace(/^\//, '') + u.search + u.hash;
      }
      return rel;
    } catch (_) {
      var hash = '';
      var hashIdx = url.indexOf('#');
      if (hashIdx >= 0) {
        hash = url.slice(hashIdx);
        url = url.slice(0, hashIdx);
      }
      var sep = url.indexOf('?') >= 0 ? '&' : '?';
      return url + sep + 'lang=' + encodeURIComponent(lang) + hash;
    }
  }

  function setLang(lang) {
    lang = normalizeLang(lang) || 'en';
    try { localStorage.setItem(STORAGE_KEY, lang); } catch (_) {}
    var url = new URL(window.location.href);
    url.searchParams.set('lang', lang);
    window.location.href = url.pathname + url.search + url.hash;
  }

  function translatedEntity(entity) {
    var lang = getLang();
    if (lang === 'en' || !entity) return entity || {};
    var overlay = entity.i18n && entity.i18n[lang] ? entity.i18n[lang] : {};
    var out = {};
    for (var key in entity) out[key] = entity[key];
    for (var k in overlay) {
      if (overlay[k]) out[k] = overlay[k];
    }
    return out;
  }

  function phaseText(phase) {
    return translatedEntity(phase);
  }

  function lessonText(lesson) {
    return translatedEntity(lesson);
  }

  function glossaryTerms() {
    if (typeof GLOSSARY === 'undefined' || !Array.isArray(GLOSSARY)) return [];
    var lang = getLang();
    if (lang === 'en' || typeof GLOSSARY_I18N === 'undefined' || !GLOSSARY_I18N[lang]) return GLOSSARY;
    var overlays = {};
    var translated = GLOSSARY_I18N[lang] || [];
    for (var i = 0; i < translated.length; i++) {
      overlays[translated[i].term] = translated[i];
    }
    return GLOSSARY.map(function (term) {
      var overlay = overlays[term.term] || {};
      return {
        term: overlay.term || term.term,
        says: overlay.says || term.says,
        means: overlay.means || term.means
      };
    });
  }

  function applyText(root) {
    root = root || document;
    var lang = getLang();
    document.documentElement.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');

    root.querySelectorAll('[data-i18n]').forEach(function (el) {
      el.textContent = t(el.getAttribute('data-i18n'));
    });
    root.querySelectorAll('[data-i18n-html]').forEach(function (el) {
      el.innerHTML = t(el.getAttribute('data-i18n-html'));
    });
    root.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      el.setAttribute('placeholder', t(el.getAttribute('data-i18n-placeholder')));
    });
    root.querySelectorAll('[data-i18n-aria]').forEach(function (el) {
      el.setAttribute('aria-label', t(el.getAttribute('data-i18n-aria')));
    });
    root.querySelectorAll('[data-i18n-title]').forEach(function (el) {
      el.setAttribute('title', t(el.getAttribute('data-i18n-title')));
    });

    root.querySelectorAll('a[href]').forEach(function (a) {
      var href = a.getAttribute('href');
      if (!href || /^(https?:|mailto:|data:|javascript:)/i.test(href) || href.charAt(0) === '#') return;
      a.setAttribute('href', withLangHref(href));
    });
  }

  function addLanguageToggle() {
    if (document.getElementById('langToggle')) return;
    var anchor = document.getElementById('themeToggle') || document.querySelector('.search-toggle');
    if (!anchor || !anchor.parentNode) return;
    var btn = document.createElement('button');
    btn.className = 'lang-toggle';
    btn.id = 'langToggle';
    btn.type = 'button';
    btn.setAttribute('aria-label', t('lang.toggleLabel'));
    btn.textContent = t('lang.toggle');
    btn.addEventListener('click', function () {
      setLang(getLang() === 'zh' ? 'en' : 'zh');
    });
    anchor.parentNode.insertBefore(btn, anchor);
  }

  function init() {
    addLanguageToggle();
    applyText(document);
  }

  window.AIFSI18N = {
    getLang: getLang,
    setLang: setLang,
    t: t,
    lessonText: lessonText,
    phaseText: phaseText,
    glossaryTerms: glossaryTerms,
    withLangHref: withLangHref,
    applyText: applyText,
    init: init
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());

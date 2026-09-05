/* Shared lifecycle for silent, inline media. Templates intentionally omit src/autoplay. */
(function () {
  'use strict';
  var players = new Map();
  var modal = null;
  var queued = false;
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  function within(el, margin) {
    if (!el.isConnected || el.closest('[hidden], .hidden-card')) return false;
    var rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0 && rect.bottom > -margin &&
      rect.top < innerHeight + margin && rect.right > 0 && rect.left < innerWidth;
  }
  function allowed(video, state) {
    return !document.hidden && (!modal || modal.contains(video)) && within(video, 0) &&
      !state.userPaused && (!reducedMotion.matches || state.manual);
  }
  function pause(video, state) {
    if (!video.paused) {
      state.internalPauses++;
      video.pause();
    }
  }
  function clearTimer(state) {
    clearTimeout(state.timer);
    state.timer = null;
  }
  function action(state, label) {
    state.button.textContent = label || '';
    state.button.hidden = !label;
    state.button.setAttribute('aria-label', (label || 'Play') + ' video');
  }
  function hydrate(video) {
    if (!video.getAttribute('src') && video.dataset.src) {
      video.src = video.dataset.src;
      video.preload = 'auto';
      video.load();
    }
  }
  function play(video, state, gesture) {
    if (state.pending || (!gesture && (!allowed(video, state) || state.blocked || state.failed))) return;
    hydrate(video);
    if (!video.getAttribute('src') || !video.paused) return;
    state.pending = true;
    var attempt = video.play();
    if (!attempt || !attempt.then) { state.pending = false; return; }
    attempt.then(function () {
      state.pending = false;
      state.blocked = false;
      action(state, null);
      if (!allowed(video, state)) pause(video, state);
      else if (video.paused) schedule();
    }).catch(function (error) {
      state.pending = false;
      if (!players.has(video)) return;
      if (error.name === 'AbortError') { schedule(); return; }
      if (error.name === 'NotAllowedError') {
        state.blocked = true;
        action(state, 'Play');
      } else {
        state.failed = true;
        action(state, 'Retry');
      }
    });
  }
  function recover(video, state) {
    if (state.timer || !allowed(video, state) || !navigator.onLine || state.retries >= 2) return;
    state.timer = setTimeout(function () {
      state.timer = null;
      if (!players.has(video) || !allowed(video, state) || !navigator.onLine) return;
      state.retries++;
      state.failed = false;
      state.retryable = false;
      pause(video, state);
      video.load();
      play(video, state, false);
    }, 1000 * Math.pow(2, state.retries));
  }
  function register(video) {
    if (players.has(video)) return;
    var host = video.parentElement;
    host.classList.add('media-host');
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'media-action';
    button.hidden = true;
    host.appendChild(button);
    var state = { button: button, retries: 0, internalPauses: 0, manual: false,
      userPaused: false, pending: false, failed: false, blocked: false, retryable: false, timer: null };
    players.set(video, state);
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.loop = true;
    button.addEventListener('click', function (event) {
      event.preventDefault();
      event.stopPropagation();
      clearTimer(state);
      state.userPaused = false;
      state.manual = true;
      state.blocked = false;
      state.retries = 0;
      if (state.failed || video.error) {
        state.failed = false;
        state.retryable = false;
        pause(video, state);
        video.load();
      }
      play(video, state, true);
    });
    button.addEventListener('keydown', function (event) { event.stopPropagation(); });
    video.addEventListener('pause', function () {
      if (state.internalPauses) { state.internalPauses--; return; }
      // Browser-driven pauses when hidden must not become a persistent user pause.
      if (within(video, 0) && !document.hidden && (!modal || modal.contains(video)) && !video.error) {
        state.userPaused = true;
        action(state, 'Play');
      }
    });
    video.addEventListener('play', function () {
      if (!state.pending) {
        state.manual = true;
        state.userPaused = false;
        state.blocked = false;
      }
      if (!allowed(video, state)) pause(video, state);
    });
    video.addEventListener('playing', function () {
      clearTimer(state);
      state.failed = false;
      state.retryable = false;
      action(state, null);
    });
    video.addEventListener('canplay', function () { play(video, state, false); });
    video.addEventListener('error', function () {
      clearTimer(state);
      state.failed = true;
      // Some engines report failed initial HTTP requests as SRC_NOT_SUPPORTED.
      state.retryable = !!video.error && (video.error.code === 2 ||
        (video.error.code === 4 && !!video.canPlayType('video/mp4')));
      action(state, 'Retry');
      if (state.retryable) recover(video, state);
    });
    function stalled() {
      if (state.timer || !allowed(video, state) || state.blocked || state.failed) return;
      state.timer = setTimeout(function () {
        state.timer = null;
        if (!allowed(video, state) || video.readyState >= 3) return;
        state.failed = true;
        state.retryable = true;
        action(state, 'Retry');
        recover(video, state);
      }, 12000);
    }
    video.addEventListener('waiting', stalled);
    video.addEventListener('stalled', stalled);
    nearObserver.observe(video);
    visibleObserver.observe(video);
  }
  function refresh() {
    queued = false;
    document.querySelectorAll('video[data-media]').forEach(register);
    document.querySelectorAll('img[data-src]').forEach(function (img) {
      nearObserver.observe(img);
      if (within(img, 300)) {
        if (img.dataset.srcset) {
          img.srcset = img.dataset.srcset;
          delete img.dataset.srcset;
        }
        img.src = img.dataset.src;
        delete img.dataset.src;
        nearObserver.unobserve(img);
      }
    });
    players.forEach(function (state, video) {
      var active = allowed(video, state);
      if (!active) {
        pause(video, state);
        clearTimer(state);
      }
      if (!document.hidden && (!modal || modal.contains(video)) && within(video, 300) &&
          (!reducedMotion.matches || state.manual)) hydrate(video);
      if (reducedMotion.matches && !state.manual && !state.failed) action(state, 'Play');
      if (active) {
        if (state.failed && state.retryable) recover(video, state);
        else play(video, state, false);
      }
    });
  }
  function schedule() {
    if (!queued) { queued = true; requestAnimationFrame(refresh); }
  }
  // Refresh at preload/play boundaries, rather than scanning on every scroll frame.
  // A small positive threshold also catches entry after an edge-only intersection.
  var nearObserver = new IntersectionObserver(schedule, { rootMargin: '300px 0px', threshold: 0.001 });
  var visibleObserver = new IntersectionObserver(schedule, { threshold: 0.001 });
  window.SiteMedia = {
    refresh: refresh,
    setModal: function (stage) { modal = stage; refresh(); },
    release: function (container) {
      container.querySelectorAll('video[data-media]').forEach(function (video) {
        var state = players.get(video);
        if (!state) return;
        clearTimer(state);
        pause(video, state);
        nearObserver.unobserve(video);
        visibleObserver.unobserve(video);
        players.delete(video);
        state.button.remove();
        video.removeAttribute('src');
        video.load();
      });
    }
  };
  document.addEventListener('visibilitychange', refresh);
  window.addEventListener('pageshow', refresh);
  window.addEventListener('online', refresh);
  reducedMotion.addEventListener('change', refresh);
  // Deferred scripts run at interactive: let initial category filters settle first.
  if (document.readyState !== 'complete') document.addEventListener('DOMContentLoaded', refresh);
  else refresh();
})();

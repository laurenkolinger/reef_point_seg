/* VICARIUS tooltip bubble: shows each control's title on hover and on
   keyboard focus, parks the title in data-tip while open so the browser
   draws no second tooltip, Escape closes. Spec section 7. */
(function () {
  'use strict';
  var bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.id = 'vicarius-bubble';
  bubble.setAttribute('role', 'tooltip');
  bubble.hidden = true;
  function mount() { if (!bubble.parentNode) document.body.appendChild(bubble); }
  function open(el) {
    var text = el.getAttribute('title');
    if (!text) return;
    mount();
    el.dataset.tip = text;
    el.removeAttribute('title');
    var existingDesc = el.getAttribute('aria-describedby');
    if (existingDesc) el.dataset.desc = existingDesc;
    el.setAttribute('aria-describedby', 'vicarius-bubble');
    bubble.textContent = text;
    bubble.hidden = false;
    var r = el.getBoundingClientRect();
    var maxLeft = window.scrollX + document.documentElement.clientWidth - bubble.offsetWidth - 8;
    bubble.style.top = (r.bottom + window.scrollY + 6) + 'px';
    bubble.style.left = Math.max(8, Math.min(r.left + window.scrollX, maxLeft)) + 'px';
  }
  function close(el) {
    if (el && el.dataset && el.dataset.tip) { el.setAttribute('title', el.dataset.tip); delete el.dataset.tip; }
    if (el && el.dataset && el.dataset.desc) { el.setAttribute('aria-describedby', el.dataset.desc); delete el.dataset.desc; }
    else if (el) { el.removeAttribute('aria-describedby'); }
    bubble.hidden = true;
  }
  document.addEventListener('mouseover', function (e) { var el = e.target.closest && e.target.closest('[title]'); if (el) open(el); });
  document.addEventListener('mouseout', function (e) { var el = e.target.closest && e.target.closest('[data-tip]'); if (el) close(el); });
  document.addEventListener('focusin', function (e) { var el = e.target.closest && e.target.closest('[title]'); if (el) open(el); });
  document.addEventListener('focusout', function (e) { var el = e.target.closest && e.target.closest('[data-tip]'); if (el) close(el); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(document.querySelector('[data-tip]')); });
  window.VicariusTooltips = { open: open, close: close };
})();

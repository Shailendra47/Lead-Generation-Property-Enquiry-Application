/**
 * Day (Light) / Night (Dark) Theme Controller for Profilent
 * Smooth eye-strain reduction with persistent user preference across all pages & tabs.
 */
(function() {
  'use strict';

  const STORAGE_KEY = 'profilent-theme';
  const LOGO_LIGHT = '/image/profilent-logo.png?v=20261027-new-logo';
  const LOGO_DARK = '/image/profilent-logo-dark.png?v=20261029-dark-logo2';

  function getSystemPreference() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function getSavedTheme() {
    return localStorage.getItem(STORAGE_KEY) || getSystemPreference();
  }

  function updateBrandLogos(isDark) {
    const targetSrc = isDark ? LOGO_DARK : LOGO_LIGHT;
    const logos = document.querySelectorAll('img.contact-brand-img, img.brand-logo, [data-brand-logo]');
    logos.forEach(img => {
      if (img.getAttribute('src') !== targetSrc) {
        img.setAttribute('src', targetSrc);
      }
    });
  }

  function applyTheme(theme) {
    const isDark = theme === 'dark';
    const root = document.documentElement;
    const body = document.body;

    if (isDark) {
      root.classList.add('dark-theme');
      root.setAttribute('data-theme', 'dark');
      if (body) {
        body.classList.add('dark-theme');
        body.setAttribute('data-theme', 'dark');
      }
    } else {
      root.classList.remove('dark-theme');
      root.setAttribute('data-theme', 'light');
      if (body) {
        body.classList.remove('dark-theme');
        body.setAttribute('data-theme', 'light');
      }
    }

    // Switch logos to Image 3 (Screenshot (1734).png) in dark mode
    updateBrandLogos(isDark);

    // Update all toggle buttons on the page
    document.querySelectorAll('.day-night-toggle').forEach(btn => {
      btn.setAttribute('aria-checked', isDark ? 'true' : 'false');
      btn.setAttribute('title', isDark ? 'Switch to Day Mode (Light)' : 'Switch to Night Mode (Dark - Eye Strain Reduction)');
    });
  }

  // 1. Immediately apply saved theme to avoid Flash of Unstyled Content (FOUC)
  applyTheme(getSavedTheme());

  // 2. Set up toggle buttons & logo swap once DOM is loaded
  function initToggles() {
    const theme = getSavedTheme();
    applyTheme(theme);
    updateBrandLogos(theme === 'dark');

    document.querySelectorAll('.day-night-toggle').forEach(btn => {
      btn.addEventListener('click', function(e) {
        e.preventDefault();
        const currentTheme = document.documentElement.classList.contains('dark-theme') ? 'dark' : 'light';
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        localStorage.setItem(STORAGE_KEY, nextTheme);
        applyTheme(nextTheme);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initToggles);
  } else {
    initToggles();
  }

  // 3. Multi-tab synchronization
  window.addEventListener('storage', function(e) {
    if (e.key === STORAGE_KEY && e.newValue) {
      applyTheme(e.newValue);
    }
  });

  // Export helper for inline calls if needed
  window.ProfilentTheme = {
    applyTheme: applyTheme,
    getTheme: getSavedTheme,
    toggle: function() {
      const current = document.documentElement.classList.contains('dark-theme') ? 'dark' : 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      localStorage.setItem(STORAGE_KEY, next);
      applyTheme(next);
      return next;
    }
  };
})();


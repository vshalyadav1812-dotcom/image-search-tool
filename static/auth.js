(() => {
  const KEY_NAME = "imageSearchApiKey";
  const protectedPaths = new Set(["/index", "/status", "/search", "/search_image", "/image"]);
  const nativeFetch = window.fetch.bind(window);
  const apiKey = () => sessionStorage.getItem(KEY_NAME) || "";

  window.fetch = (input, init = {}) => {
    const requestUrl = new URL(typeof input === "string" ? input : input.url, window.location.href);
    const headers = new Headers(init.headers || (input instanceof Request ? input.headers : undefined));
    if (requestUrl.origin === window.location.origin && protectedPaths.has(requestUrl.pathname)) {
      const key = apiKey();
      if (key) headers.set("X-API-Key", key);
      return nativeFetch(input, { ...init, headers });
    }
    return nativeFetch(input, init);
  };

  function addKeyPanel() {
    if (document.getElementById("api-key-panel")) return;
    const panel = document.createElement("section");
    panel.id = "api-key-panel";
    panel.style.cssText = "max-width:900px;margin:0 auto 24px;padding:16px;border:1px solid #64748b;border-radius:10px;background:#111827;color:#f8fafc;font-family:system-ui";
    panel.innerHTML = '<label for="api-key-input" style="display:block;margin-bottom:8px;font-weight:600">Local API key</label>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap"><input id="api-key-input" type="password" autocomplete="current-password" placeholder="Enter IMAGE_SEARCH_API_KEY" style="flex:1;min-width:220px;padding:10px;border-radius:6px;border:1px solid #64748b;background:#0f172a;color:white">' +
      '<button id="api-key-save" type="button" style="width:auto;padding:10px 16px">Connect</button></div>' +
      '<p id="api-key-message" role="status" style="margin-top:8px;font-size:.9rem">Key is kept in this tab session only.</p>';
    (document.querySelector(".container") || document.body).prepend(panel);
    const input = panel.querySelector("#api-key-input");
    const message = panel.querySelector("#api-key-message");
    input.value = apiKey();
    panel.querySelector("#api-key-save").addEventListener("click", async () => {
      const value = input.value.trim();
      if (!value) {
        message.textContent = "Enter the API key configured on the server.";
        return;
      }
      try {
        const response = await nativeFetch("/status", { headers: { "X-API-Key": value } });
        if (!response.ok) {
          message.textContent = response.status === 503
            ? "Server authentication is not configured. Set IMAGE_SEARCH_API_KEY and restart."
            : "Authentication failed. Check the key and try again.";
          return;
        }
        sessionStorage.setItem(KEY_NAME, value);
        message.textContent = "Connected. API requests are authenticated.";
        window.dispatchEvent(new CustomEvent("image-search-authenticated"));
      } catch (_) {
        message.textContent = "Could not connect to the server.";
      }
    });
  }

  async function protectImage(img) {
    if (img.dataset.authLoading || !img.src) return;
    const url = new URL(img.src, window.location.href);
    if (url.origin !== window.location.origin || url.pathname !== "/image") return;
    img.dataset.authLoading = "1";
    try {
      const response = await nativeFetch(url, { headers: { "X-API-Key": apiKey() } });
      if (!response.ok) {
        img.alt = response.status === 401 ? "Enter API key to view this image" : "Image unavailable";
        return;
      }
      const blobUrl = URL.createObjectURL(await response.blob());
      img.dataset.authBlobUrl = blobUrl;
      img.src = blobUrl;
    } catch (_) {
      img.alt = "Image unavailable";
    }
  }

  function watchImages() {
    document.querySelectorAll('img[src^="/image"]').forEach(protectImage);
    new MutationObserver(records => records.forEach(record => {
      if (record.type === "attributes" && record.target instanceof HTMLImageElement) protectImage(record.target);
      for (const node of record.addedNodes || []) {
        if (node instanceof HTMLImageElement) protectImage(node);
        if (node.querySelectorAll) node.querySelectorAll("img").forEach(protectImage);
      }
    })).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
    window.addEventListener("beforeunload", () => {
      document.querySelectorAll("img[data-auth-blob-url]").forEach(img => URL.revokeObjectURL(img.dataset.authBlobUrl));
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => { addKeyPanel(); watchImages(); });
  } else {
    addKeyPanel();
    watchImages();
  }
})();

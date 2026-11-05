const OPTIONS = {
  planning_to_buy: [
    "Immediately", "Within 1 Month", "Within 3 Months", "Within 6 Months",
    "Within 12 Months", "More Than 1 Year", "Just Exploring / Researching",
  ],
  budget_range: [
    "Up to 25 lakh", "25 lakh - 50 lakh", "50 lakh - 1 crore",
    "1 crore - 2 crore", "Above 2 crore",
  ],
  payment_option: [
    "Bank Finance with EMI", "Full Cash Payment", "Home Loan (Mortgage)",
    "Down Payment Plan", "Construction-Linked Payment Plan (CLP)",
    "Possession-Linked Payment Plan (PLP)", "Installment Payment Plan",
    "Developer Financing", "Rent-to-Own Scheme", "Part Cash + Home Loan",
    "Part Payment with Balance at Registration", "Flexible Payment Plan",
    "Subvention Scheme (Developer Pays Pre-EMI)",
    "Joint Home Loan (Co-applicant Financing)",
    "NRI Payment Plan (for Overseas Buyers)",
  ],
  bhk_preference: [
    "1 RK", "1 BHK", "2 BHK", "2.5 BHK", "3 BHK", "3.5 BHK",
    "4 BHK", "5 BHK+", "Villa", "Penthouse",
  ],
};

const form = document.getElementById("enquiryForm");
const submitBtn = document.getElementById("submitBtn");
const mobile = document.getElementById("mobile_number");
const errorPopup = document.getElementById("errorPopup");
const errorMessage = document.getElementById("errorMessage");
const errorTitle = document.getElementById("errorTitle");
let fieldToFocusAfterError = null;

// Defensive cleanup: ensure no legacy popup dialog exists in the DOM
const legacySuccessPopup = document.getElementById("successPopup");
if (legacySuccessPopup) legacySuccessPopup.remove();
document.querySelectorAll(".popup-overlay:not(.alert-overlay)").forEach((el) => el.remove());

function hideErrorPopup() {
  if (errorPopup) errorPopup.hidden = true;
  if (fieldToFocusAfterError) fieldToFocusAfterError.focus();
}
const errorRetryBtn = document.getElementById("errorRetry");
if (errorRetryBtn) errorRetryBtn.addEventListener("click", hideErrorPopup);
if (errorPopup) {
  errorPopup.addEventListener("click", (e) => { if (e.target === errorPopup) hideErrorPopup(); });
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && errorPopup && !errorPopup.hidden) {
    hideErrorPopup();
  }
});

function fieldLabel(fieldName) {
  const labels = {
    first_name: "First name",
    last_name: "Last name",
    email: "Email",
    mobile_number: "Mobile number",
    location_preference: "Location preference",
    planning_to_buy: "Planning to buy",
    budget_range: "Budget range",
    payment_option: "Payment option",
    bhk_preference: "BHK preference",
    remarks_requirements: "Remarks / requirements",
  };
  return labels[fieldName] || "Field";
}

function presentApiError(data, status) {
  fieldToFocusAfterError = null;
  const detail = data?.detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      const location = Array.isArray(item.loc) ? item.loc.at(-1) : "";
      if (location && location !== "body") fieldToFocusAfterError = form.elements.namedItem(location);
      const reason = item.msg === "Field required" ? "is required" : item.msg;
      return `${fieldLabel(location)} ${reason}.`;
    });
    return messages.join(" ");
  }
  if (typeof detail === "string") {
    if (/location/i.test(detail)) fieldToFocusAfterError = form.elements.namedItem("location_preference");
    else if (/mobile|phone/i.test(detail)) fieldToFocusAfterError = form.elements.namedItem("mobile_number");
    else if (/email/i.test(detail)) fieldToFocusAfterError = form.elements.namedItem("email");
    return detail;
  }
  return status === 422
    ? "Some information could not be validated. Please check your entries and try again."
    : "We could not save your enquiry right now. Please try again in a moment.";
}

function showError(message, title = "CHECK YOUR DETAILS", field = null) {
  if (errorTitle) errorTitle.textContent = title;
  errorMessage.textContent = message;
  fieldToFocusAfterError = field;
  errorPopup.hidden = false;
  document.getElementById("errorRetry").focus();
}

// Fill the dropdowns
for (const [id, list] of Object.entries(OPTIONS)) {
  const select = document.getElementById(id);
  select.add(new Option("Select an option", ""));
  list.forEach((item) => select.add(new Option(item, item)));
}

// Mobile: optional +, then 7-15 digits (spaces and hyphens allowed)
function checkMobile() {
  const digits = mobile.value.replace(/[\s-]/g, "");
  mobile.setCustomValidity(/^\+?[0-9]{7,15}$/.test(digits) ? "" : "invalid");
}
mobile.addEventListener("input", checkMobile);

// Clean accidental stray quotes/backticks on blur
document.querySelectorAll("input").forEach((inp) => {
  inp.addEventListener("blur", () => {
    if (inp.type === "email" || inp.name === "email" || inp.name === "first_name" || inp.name === "last_name") {
      inp.value = inp.value.replace(/[`'"]/g, "").trim();
    }
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Auto-clean stray backticks, quotes, or whitespace before validity check
  const emailInput = form.elements.namedItem("email");
  if (emailInput) {
    emailInput.value = emailInput.value.replace(/[`'"]/g, "").trim();
  }
  const fnInput = form.elements.namedItem("first_name");
  if (fnInput) {
    fnInput.value = fnInput.value.replace(/[`'"]/g, "").trim();
  }
  const lnInput = form.elements.namedItem("last_name");
  if (lnInput) {
    lnInput.value = lnInput.value.replace(/[`'"]/g, "").trim();
  }

  checkMobile();

  if (!form.checkValidity()) {
    form.classList.add("was-validated");
    const invalidField = form.querySelector(":invalid");
    if (invalidField) {
      invalidField.focus();
      let label = invalidField.previousElementSibling ? invalidField.previousElementSibling.textContent.replace("*", "").trim() : "field";
      showError(`Please fill in or correct the ${label} before submitting.`, "CHECK YOUR DETAILS", invalidField);
    }
    return;
  }

  const payload = {};
  new FormData(form).forEach((value, key) => (payload[key] = value.trim()));

  // Strict email validation: uppercase/capitalized letters are prohibited; lowercase, numbers, @ required
  const rawEmail = (payload.email || "").trim();
  if (/[A-Z]/.test(rawEmail)) {
    showError("Email must be strictly lowercase (lowercase letters, numbers, and @ sign only, no uppercase or capitalized letters).", "INVALID EMAIL FORMAT", emailInput);
    if (emailInput) emailInput.focus();
    return;
  }
  const lowercaseEmailPattern = /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/;
  if (!lowercaseEmailPattern.test(rawEmail)) {
    showError("Please enter a valid lowercase email address with numbers and @ sign (e.g. name123@gmail.com).", "INVALID EMAIL FORMAT", emailInput);
    if (emailInput) emailInput.focus();
    return;
  }

  // Merge country code with mobile number into E.164 international format
  const countryCode = form.elements.namedItem("country_code")?.value || "+91";
  let rawNumber = (payload.mobile_number || "").replace(/[\s-]/g, "");
  const digitsOnly = rawNumber.replace(/\D/g, "").replace(/^0+/, "");
  if (countryCode === "+91") {
    if (!/^[6-9]\d{9}$/.test(digitsOnly)) {
      showError("Please enter a valid 10-digit Indian mobile number starting with 6, 7, 8, or 9.");
      const phoneInput = form.elements.namedItem("mobile_number");
      if (phoneInput) phoneInput.focus();
      return;
    }
  } else if (digitsOnly.length < 7 || digitsOnly.length > 15) {
    showError("Please enter a valid mobile number (7 to 15 digits).");
    const phoneInput = form.elements.namedItem("mobile_number");
    if (phoneInput) phoneInput.focus();
    return;
  }

  payload.mobile_number = rawNumber.startsWith("+") ? `+${digitsOnly}` : `${countryCode}${digitsOnly}`;
  delete payload.country_code;

  const optInEl = form.elements.namedItem("whatsapp_opt_in") || document.getElementById("whatsapp_opt_in");
  payload.whatsapp_opt_in = optInEl ? Boolean(optInEl.checked) : true;

  submitBtn.disabled = true;
  submitBtn.textContent = "Submitting...";

  try {
    const res = await fetch("/api/enquiry", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (res.ok) {
      const waUrl = data.whatsapp_url;
      if (payload.whatsapp_opt_in && waUrl) {
        submitBtn.disabled = false;
        submitBtn.setAttribute("type", "button");
        submitBtn.innerHTML = '<span style="display:inline-flex;align-items:center;justify-content:center;gap:8px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12.04 2c-5.46 0-9.91 4.45-9.91 9.91 0 1.75.46 3.45 1.32 4.95L2.05 22l5.25-1.38c1.45.79 3.08 1.21 4.74 1.21 5.46 0 9.91-4.45 9.91-9.91 0-2.65-1.03-5.14-2.9-7.01A9.816 9.816 0 0 0 12.04 2z"/></svg> Redirecting to WhatsApp...</span>';
        submitBtn.style.background = "#25D366";
        submitBtn.style.color = "#ffffff";
        submitBtn.onclick = (ev) => {
          ev.preventDefault();
          window.location.href = waUrl;
        };

        // Guaranteed redirect that bypasses browser async popup blockers
        const isMobile = /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        if (isMobile) {
          window.location.href = waUrl;
        } else {
          let openedWin = null;
          try {
            openedWin = window.open(waUrl, "_blank");
          } catch (_) {
            openedWin = null;
          }
          if (!openedWin || openedWin.closed || typeof openedWin.closed === "undefined") {
            window.location.href = waUrl;
          }
        }
      } else {
        submitBtn.disabled = true;
        submitBtn.textContent = "✓ Enquiry Submitted Successfully!";
      }
      setTimeout(() => {
        form.reset();
        submitBtn.disabled = false;
        submitBtn.setAttribute("type", "submit");
        submitBtn.textContent = "Submit Enquiry";
        submitBtn.style.background = "";
        submitBtn.style.color = "";
        submitBtn.onclick = null;
        const optIn = document.getElementById("whatsapp_opt_in");
        if (optIn) optIn.checked = true;
      }, 7000);
      return;
    } else if (res.status === 422) {
      showError(presentApiError(data, 422), "CHECK YOUR DETAILS");
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit Enquiry";
    } else {
      showError(presentApiError(data, res.status), "UNABLE TO SUBMIT");
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit Enquiry";
    }
  } catch (err) {
    showError("Could not reach the server. Please check your internet connection and try again.", "CONNECTION ERROR");
    submitBtn.disabled = false;
    submitBtn.textContent = "Submit Enquiry";
  }
});

// ---- Location suggestions while typing (OpenStreetMap data via the free Photon service)
const locInput = document.getElementById("location_preference");
const locList = document.getElementById("locList");
let locTimer, locAbort, locItems = [], locActive = -1;

function hideLocations() {
  locList.hidden = true;
  locActive = -1;
}

function showLocMessage(text) {
  locItems = [];
  locActive = -1;
  locList.innerHTML = "";
  const li = document.createElement("li");
  li.className = "muted";
  li.textContent = text;
  locList.appendChild(li);
  locList.hidden = false;
}

function renderLocations() {
  if (!locItems.length) {
    showLocMessage("No suggestions found. You can enter the place name and it will be verified when you submit.");
    return;
  }
  locList.innerHTML = "";
  locActive = -1;
  locItems.forEach((label, i) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.dataset.i = i;
    li.textContent = label;
    locList.appendChild(li);
  });
  locList.hidden = false;
}

function pickLocation(i) {
  locInput.value = locItems[i];
  hideLocations();
}

function setActive(n) {
  const opts = locList.querySelectorAll("li[data-i]");
  if (!opts.length) return;
  locActive = (n + opts.length) % opts.length;
  opts.forEach((o, idx) => o.classList.toggle("active", idx === locActive));
  opts[locActive].scrollIntoView({ block: "nearest" });
}

async function searchLocations(q) {
  if (locAbort) locAbort.abort();
  locAbort = new AbortController();
  showLocMessage("Searching...");
  const params = new URLSearchParams({ q });
  try {
    const res = await fetch("/api/locations?" + params, { signal: locAbort.signal });
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();
    locItems = data.locations || [];
    renderLocations();
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error("Location search failed:", err);
    showLocMessage("Suggestions are unavailable. You can still submit a place name for verification.");
  }
}

locInput.addEventListener("input", () => {
  clearTimeout(locTimer);
  const q = locInput.value.trim();
  if (q.length < 3) { hideLocations(); return; }
  locTimer = setTimeout(() => searchLocations(q), 350);
});
locInput.addEventListener("keydown", (e) => {
  if (locList.hidden) return;
  if (e.key === "ArrowDown") { e.preventDefault(); setActive(locActive + 1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); setActive(locActive - 1); }
  else if (e.key === "Enter" && locActive >= 0) { e.preventDefault(); pickLocation(locActive); }
  else if (e.key === "Escape") { hideLocations(); }
});
locInput.addEventListener("blur", hideLocations);
locList.addEventListener("mousedown", (e) => {
  const li = e.target.closest("li[data-i]");
  if (li) { e.preventDefault(); pickLocation(Number(li.dataset.i)); }
});




const form = document.getElementById("contactForm");
const submitBtn = document.getElementById("submitBtn");
const errorPopup = document.getElementById("errorPopup");
const errorMessage = document.getElementById("errorMessage");
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

// Ensure WhatsApp consent checkbox is checked by default
function resetOptInCheckbox() {
  const optIn = document.getElementById("whatsapp_opt_in");
  if (optIn) optIn.checked = true;
}
resetOptInCheckbox();
window.addEventListener("pageshow", resetOptInCheckbox);

function fieldLabel(fieldName) {
  const labels = {
    full_name: "Full name", FirstName: "First Name", "Last Name": "Last Name",
    email: "Email", Email: "Email",
    contact_number: "Phone number", PhoneNumber: "Phone number",
    whatsapp_opt_in: "WhatsApp preference",
  };
  return labels[fieldName] || "Form";
}

function presentApiError(data, status) {
  fieldToFocusAfterError = null;
  const detail = data?.detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      const location = Array.isArray(item.loc) ? item.loc.at(-1) : "";
      if (location && location !== "body") {
        fieldToFocusAfterError = form.elements.namedItem(location) ||
                                 form.elements.namedItem("FirstName") ||
                                 form.elements.namedItem("Email") ||
                                 form.elements.namedItem("PhoneNumber");
      }
      const reason = item.msg === "Field required" ? "is required" : item.msg;
      return `${fieldLabel(location)} ${reason}.`;
    });
    return messages.join(" ");
  }
  if (typeof detail === "string") {
    const match = detail.match(/(?:Enter|Please enter) (?:the )?(?:WhatsApp )?number/i);
    if (match) {
      fieldToFocusAfterError = form.elements.namedItem("PhoneNumber") || form.elements.namedItem("contact_number");
    }
    return detail;
  }
  return status === 422
    ? "Some information could not be validated. Please check your entries and try again."
    : "We could not send your enquiry right now. Please try again in a moment.";
}

function showError(message) {
  errorMessage.textContent = message;
  errorPopup.hidden = false;
  document.getElementById("errorRetry").focus();
}

// Clean accidental stray quotes/backticks on blur and validate lowercase email
const emailInputEl = form.elements.namedItem("Email") || form.elements.namedItem("email");
if (emailInputEl) {
  emailInputEl.addEventListener("input", () => {
    if (/[A-Z]/.test(emailInputEl.value)) {
      emailInputEl.setCustomValidity("Email must be entirely in lowercase. Capital letters are not allowed.");
    } else {
      emailInputEl.setCustomValidity("");
    }
  });
}

document.querySelectorAll("input").forEach((inp) => {
  inp.addEventListener("blur", () => {
    if (inp.type === "email" || inp.name === "Email" || inp.name === "FirstName" || inp.name === "Last Name") {
      inp.value = inp.value.replace(/[`'"]/g, "").trim();
    }
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Auto-clean accidental backticks, quotes, or stray spaces before checking validity
  const emailInput = form.elements.namedItem("Email") || form.elements.namedItem("email");
  if (emailInput) {
    emailInput.value = emailInput.value.replace(/[`'"]/g, "").trim();
  }
  const fnInput = form.elements.namedItem("FirstName") || form.elements.namedItem("first_name");
  if (fnInput) {
    fnInput.value = fnInput.value.replace(/[`'"]/g, "").trim();
  }
  const lnInput = form.elements.namedItem("Last Name") || form.elements.namedItem("LastName") || form.elements.namedItem("last_name");
  if (lnInput) {
    lnInput.value = lnInput.value.replace(/[`'"]/g, "").trim();
  }
  const phoneInput = form.elements.namedItem("PhoneNumber") || form.elements.namedItem("contact_number");
  if (phoneInput) {
    phoneInput.value = phoneInput.value.trim();
  }

  if (!form.checkValidity()) {
    form.classList.add("was-validated");
    const firstInvalid = form.querySelector(":invalid");
    if (firstInvalid) {
      firstInvalid.focus();
      let msg = "Please fill in all required fields.";
      if (firstInvalid.type === "email" || firstInvalid.name === "Email") {
        msg = "Please enter a valid email address without any special symbols.";
      } else if (firstInvalid.name === "FirstName" || firstInvalid.name === "first_name") {
        msg = "Please enter your first name.";
      } else if (firstInvalid.name === "PhoneNumber" || firstInvalid.name === "contact_number") {
        msg = "Please enter your mobile phone number.";
      }
      showError(msg, "CHECK YOUR DETAILS");
    }
    return;
  }

  const formData = new FormData(form);
  const rawPayload = Object.fromEntries(formData.entries());

  // Extract first_name and last_name separately, and compute full_name
  let fn = (rawPayload.FirstName || rawPayload.first_name || "").trim();
  let ln = (rawPayload.LastName || rawPayload["Last Name"] || rawPayload.last_name || "").trim();
  let fullName = (rawPayload.full_name || "").trim();

  if (!fullName) {
    fullName = `${fn} ${ln}`.trim();
  } else if (!fn) {
    const parts = fullName.split(/\s+/);
    fn = parts[0] || "";
    ln = parts.slice(1).join(" ") || "";
  }
  if (!fullName && !fn) {
    showError("Please enter your name.");
    return;
  }

  const email = (rawPayload.email || rawPayload.Email || "").trim();
  if (!email) {
    showError("Please enter a valid email address.");
    return;
  }

  // Strict email validation: uppercase/capitalized letters are prohibited; lowercase, numbers, @ required
  if (/[A-Z]/.test(email)) {
    showError("Email must be strictly lowercase (lowercase letters, numbers, and @ sign only, no uppercase or capitalized letters).");
    if (emailInputEl) emailInputEl.focus();
    return;
  }

  const lowercaseEmailPattern = /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/;
  if (!lowercaseEmailPattern.test(email)) {
    showError("Please enter a valid lowercase email address with numbers and @ sign (e.g. name123@gmail.com).");
    if (emailInputEl) emailInputEl.focus();
    return;
  }

  // Merge country code with phone number into E.164 international format
  const countryCode = form.elements.namedItem("country_code")?.value || "+91";
  let rawNumber = (rawPayload.contact_number || rawPayload.PhoneNumber || "").trim().replace(/[\s-]/g, "");
  if (!rawNumber) {
    showError("Please enter your phone number.");
    return;
  }

  // Strict mobile number validation
  const digitsOnly = rawNumber.replace(/\D/g, "").replace(/^0+/, "");
  if (countryCode === "+91") {
    if (!/^[6-9]\d{9}$/.test(digitsOnly)) {
      showError("Please enter a valid 10-digit Indian mobile number starting with 6, 7, 8, or 9.");
      const phoneInput = form.elements.namedItem("PhoneNumber") || form.elements.namedItem("contact_number");
      if (phoneInput) phoneInput.focus();
      return;
    }
  } else if (digitsOnly.length < 7 || digitsOnly.length > 15) {
    showError("Please enter a valid mobile number (7 to 15 digits).");
    const phoneInput = form.elements.namedItem("PhoneNumber") || form.elements.namedItem("contact_number");
    if (phoneInput) phoneInput.focus();
    return;
  }

  let finalNumber = rawNumber.startsWith("+") ? `+${digitsOnly}` : `${countryCode}${digitsOnly}`;

  const reqValue = (rawPayload.requirement || rawPayload["What’s on your mind?"] || form.querySelector("textarea")?.value || "").trim();

  // Automatic WhatsApp opt-in (checkbox removed as form redirects to WhatsApp automatically)
  const optInEl = form.elements.namedItem("whatsapp_opt_in") || document.getElementById("whatsapp_opt_in");
  const whatsappOptIn = optInEl ? Boolean(optInEl.checked) : true;

  const payload = {
    first_name: fn,
    last_name: ln,
    full_name: fullName,
    email: email,
    contact_number: finalNumber,
    requirement: reqValue,
    whatsapp_opt_in: whatsappOptIn,
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "Submitting...";

  try {
    const res = await fetch("/api/contact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (res.ok) {
      const waUrl = data.whatsapp_url;
      if (whatsappOptIn && waUrl) {
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
        submitBtn.textContent = "✓ Submitted Successfully!";
      }
      setTimeout(() => {
        form.reset();
        submitBtn.disabled = false;
        submitBtn.setAttribute("type", "submit");
        submitBtn.textContent = "Submit";
        submitBtn.style.background = "";
        submitBtn.style.color = "";
        submitBtn.onclick = null;
        resetOptInCheckbox();
      }, 7000);
      return;
    } else {
      showError(presentApiError(data, res.status));
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit";
    }
  } catch (err) {
    showError("We couldn't reach the server. Check your connection and try submitting again.");
    submitBtn.disabled = false;
    submitBtn.textContent = "Submit";
  }
});

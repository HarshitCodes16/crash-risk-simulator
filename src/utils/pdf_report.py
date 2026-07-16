"""
PDF Report Generator
----------------------
Builds a simple, professional-looking PDF summary of a single crash-risk
scenario, using fpdf2 (pure Python, no compiled/system dependencies - chosen
specifically to avoid the kind of cross-environment compatibility issues
this project has already hit once with scikit-learn pickles).

All text is passed through _sanitize() before being written, since fpdf2's
default Helvetica font only supports Latin-1 - characters like em-dashes or
arrows (commonly produced by "smart" formatting) would otherwise raise a
silent encoding failure at PDF-generation time.
"""

from datetime import datetime
from io import BytesIO
from fpdf import FPDF, XPos, YPos

_REPLACEMENTS = {
    "\u2014": "-",    # em dash —
    "\u2013": "-",    # en dash –
    "\u2192": "->",   # rightwards arrow →
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u2026": "...",  # ellipsis …
}


def _sanitize(text):
    text = str(text)
    for bad, good in _REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Final safety net: force anything still non-Latin-1 to a safe substitute
    # instead of letting fpdf2 raise an opaque encoding error.
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_scenario_pdf(scenario, results):
    """
    scenario: dict with keys - mode, speed_label, speed_kmh, distance,
              weather, road_type, time_of_day, vehicle_type
    results:  dict with keys - crash_probability, severity (or None),
              verification_status, biggest_factor, recommendations (list[str]),
              safe_speed_kmh (or None), safe_distance (or None),
              chart_image_bytes (PNG bytes, optional)

    Returns the PDF as raw bytes, ready for st.download_button.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _sanitize("Crash Risk Assessment Report"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, _sanitize(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize(f"Scenario: {scenario['mode']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    rows = [
        (f"{scenario['speed_label']}", f"{scenario['speed_kmh']:.0f} km/h"),
        ("Following distance", f"{scenario['distance']:.0f} m"),
        ("Weather", scenario["weather"]),
        ("Road type", scenario["road_type"]),
        ("Time of day", scenario["time_of_day"]),
        ("Vehicle type", scenario["vehicle_type"]),
    ]
    for label, value in rows:
        pdf.cell(60, 6, _sanitize(f"{label}:"), border=0)
        pdf.cell(0, 6, _sanitize(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize("Risk Assessment"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(60, 6, _sanitize("Crash probability:"))
    pdf.cell(0, 6, _sanitize(f"{results['crash_probability']*100:.0f}%"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if results.get("severity") is not None:
        pdf.cell(60, 6, _sanitize("Predicted severity:"))
        pdf.cell(0, 6, _sanitize(f"{results['severity']:,.0f} N"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 6, _sanitize("Physics verification:"))
    pdf.cell(0, 6, _sanitize(results["verification_status"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(60, 6, _sanitize("Biggest risk factor:"))
    pdf.cell(0, 6, _sanitize(results["biggest_factor"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # --- Safe boundaries ---
    if results.get("safe_speed_kmh") is not None or results.get("safe_distance") is not None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _sanitize("Safe Boundaries"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        if results.get("safe_speed_kmh") is not None:
            pdf.cell(0, 6, _sanitize(
                f"Keep {scenario['speed_label'].lower()} under {results['safe_speed_kmh']:.0f} km/h to stay safe."
            ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if results.get("safe_distance") is not None:
            pdf.cell(0, 6, _sanitize(
                f"Keep at least {results['safe_distance']:.0f} m of following distance to stay safe."
            ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # --- Recommendations ---
    if results.get("recommendations"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _sanitize("Recommendations"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        for rec in results["recommendations"]:
            pdf.multi_cell(0, 6, _sanitize(f"- {rec}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # --- Risk contribution chart (static image, embedded) ---
    if results.get("chart_image_bytes"):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _sanitize("Risk Contribution by Factor"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        img_buf = BytesIO(results["chart_image_bytes"])
        pdf.image(img_buf, w=170)
        pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(
        0, 5,
        _sanitize(
            "This is a what-if scenario simulation, not a live vehicle tracking "
            "system. Generated by Crash Risk Simulator."
        ),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return output.encode("latin1")
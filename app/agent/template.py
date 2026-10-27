"""The fixed notice template. The LLM writes only {reasons_paragraph}; every legal part
(action taken, ECOA notice, creditor and agency contact) is fixed text here.

ECOA notice wording follows 12 CFR 1002.9(b)(1) and Appendix C, Form C-1. The creditor is
fictional; the agency named is the CFPB, as listed in Reg B Appendix A for creditors it
supervises. Simplified for a demo, not legal advice: a real notice also needs the FCRA
section 615 disclosures when consumer-report information was used.
"""

from datetime import date

CREDITOR = "CreditShield Lending (demo)"

NOTICE_TEMPLATE = """\
{creditor}
Date: {notice_date}
To: Applicant #{applicant_id}

Re: Your application for a {product}

Thank you for your recent application. We regret that we are unable to approve your
request for a {product}.

Principal reason(s) for our decision:
{reasons_paragraph}

[Consumer reporting agency disclosure (FCRA section 615): not included in this demo.]

NOTICE: The Federal Equal Credit Opportunity Act prohibits creditors from discriminating
against credit applicants on the basis of race, color, religion, national origin, sex,
marital status, age (provided the applicant has the capacity to enter into a binding
contract); because all or part of the applicant's income derives from any public assistance
program; or because the applicant has in good faith exercised any right under the Consumer
Credit Protection Act. The Federal agency that administers compliance with this law
concerning this creditor is the Bureau of Consumer Financial Protection, 1700 G Street NW,
Washington, DC 20552.

If you have questions about this letter, contact {creditor} at notices@example.com.
"""

PRODUCT_NAMES = {"Cash loans": "cash loan", "Revolving loans": "revolving line of credit"}


def product_name(product: str) -> str:
    return PRODUCT_NAMES.get(product, product.lower())


def render_notice(
    *, applicant_id: int, product: str, notice_date: date, reasons_paragraph: str
) -> str:
    return NOTICE_TEMPLATE.format(
        creditor=CREDITOR,
        notice_date=notice_date.isoformat(),
        applicant_id=applicant_id,
        product=product_name(product),
        reasons_paragraph=reasons_paragraph.strip(),
    )


def template_for_agent() -> str:
    """What get_notice_template returns: the letter with only the LLM's slot left open."""
    return NOTICE_TEMPLATE.replace("{reasons_paragraph}", "<<YOUR PARAGRAPH GOES HERE>>")

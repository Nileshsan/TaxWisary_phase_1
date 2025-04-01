import re
import torch
from io import BytesIO
from django.http import JsonResponse, HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load LLM model and tokenizer
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)

def get_llm_response(user_input, system_prompt):
    """
    Gets response from LLM based on user input and system prompt.
    Uses max_new_tokens instead of max_length to avoid input length issues.
    """
    user_input = user_input.strip()
    if not user_input:
        return "No input detected. Please provide your response."

    # Construct the prompt
    prompt = f"<s>[SYSTEM] {system_prompt}\n[USER] {user_input}\n[ASSISTANT]"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)  # Move inputs to GPU

    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,  # Changed from max_length to max_new_tokens
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the response on the GPU
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("[ASSISTANT]")[-1].strip()

        if not response:
            return "I'm sorry, I couldn't understand that. Could you rephrase?"
        
        return response

    except Exception as e:
        print(f"ERROR in get_llm_response: {str(e)}")
        return "An error occurred. Please try again."

def chatbot_view(request):
    """
    Django view to handle chatbot queries and return structured extracted data.
    """
    user_input = request.GET.get("query", "").strip()

    if not user_input:
        return JsonResponse({"error": "No input provided"}, status=400)

    extracted_info = extract_data(user_input)

    return JsonResponse({"response": extracted_info})

def generate_pdf(data):
    """
    Generates a PDF from extracted data and returns it as an HTTP response.
    """

    # Ensure data is valid
    if not data:
        return HttpResponse("No data provided for PDF generation.", status=400)

    try:
        # Load template and render with data
        template = get_template('users/pdf_template.html')

        html_content = template.render({'data': data})

        # Convert HTML to PDF
        pdf_result = BytesIO()
        pdf = pisa.CreatePDF(html_content, dest=pdf_result)

        if pdf.err:
            return HttpResponse("Failed to generate PDF.", status=500)

        # Prepare response with PDF
        response = HttpResponse(pdf_result.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="tax_report.pdf"'
        return response

    except Exception as e:
        return HttpResponse(f"Error generating PDF: {str(e)}", status=500)

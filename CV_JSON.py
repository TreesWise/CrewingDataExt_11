# import fitz  # PyMuPDF
from PIL import Image
import openai
from fastapi.responses import JSONResponse
import traceback
import io
import json
import fitz  # PyMuPDF
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
# from config import api_key
from dotenv import load_dotenv
import logging
import os


load_dotenv()

OPENAI_API_KEY = os.getenv("api_key")

def cv_json(file):

    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    langs = ["en"]
    all_predictions = []
    try:
        print('pdf is uploading')
        pdf_content = file.file.read() if hasattr(file, 'file') else file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        print('pdf uploaded')
        # pdf_document = fitz.open(PDF_PATH)  
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)  
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            predictions = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor)
            all_predictions.append(predictions)
        pdf_document.close()
        print('pdf contents extracted')
    except Exception as e:
        error_message = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_message)
        return JSONResponse(status_code=500, content={"status": "error", "message": error_message})


    openai.api_key = OPENAI_API_KEY
    def get_completion(prompt, model='chatgpt-4o-latest'):  
        messages = [{"role":"user", "content": prompt}]
        response = openai.ChatCompletion.create(model=model,messages=messages,temperature=0)
        return response.choices[0].message#['content']


    # Path to the JSON file
    file_path_json = r"D:\OneDrive - MariApps Marine Solutions Pte.Ltd\CV_json\CURICULUM VITAE ARIP highlighted masked.json"

    # Load the JSON file
    with open(file_path_json, 'r') as file:
        sample_output = json.load(file)

    json_template_str = json.dumps(sample_output, indent=4)

    def generate_prompt(all_predictions, json_template_str):
        prompt = f"""
            You are an expert in data extraction and JSON formatting. Your task is to extract, organize, and format the information from the provided resume content, `{all_predictions}`, into the exact JSON structure defined in `{json_template_str}`. Follow these detailed instructions carefully to ensure accuracy and completeness:

            ### **Instructions:**

            1. **JSON Structure Compliance:**
            - Your output must strictly adhere to the structure provided in `{json_template_str}`. Every key in the JSON template should be present in the final output, even if some values are `null`.
            - Ensure all sections (`basic_details`, `certificate_table`, `experience_table`, etc.) are fully populated according to the template.
            - Ensure that the output format strictly mirrors the structure of the input, with the first element in every table consistently serving as the headers. Maintain the same arrangement of columns, and ensure that all subsequent data follows this exact format, preserving the integrity of the input structure without altering the order or adding any extra details.
            

            2. **Data Extraction Requirements:**
            - **`basic_details`:**
                - Extract `City`, `State`, `Country`, and `Zipcode` from the address and map them to their respective fields.
                - Split the address correctly into `Address1`, `Address2`, `Address3`, and `Address4`.
            - **`experience_table`:**
                - Capture all details related to experience entries without omission. Merge any multi-line entries into a single continuous string, ensuring completeness.
                - Verify that fields such as `VesselName`, `VesselType`, `Company`, `EngineDetails`, `BHP`, `Rank`, and `Dates` are extracted accurately and combined correctly if split across multiple lines.
                - Maintain the original sequence and include all entries, even those with sparse data. If any field is missing, set it explicitly to `null`.
                - Ensure the correct mapping of `TEU` and `IMO` values. The `TEU` column should capture numerical values representing container capacity (e.g., 11888), while the `IMO` field should only contain a 7-digit identifier (e.g., 1234567). Make sure these fields are not interchanged. Ensure that `IMO` contains a 7-digit number. If there is no such 7 digit `IMO` number keep it as null.
                - Extract the Flag value, ensuring it is a valid country or state name (e.g., 'Panama', 'Marshall Islands'). Do not include abbreviations or incomplete entries (e.g., 'fg'). If no valid country or state is found, set the Flag field to null.
                - Don't put engine details in the `VesselSubType` part. If the is no specific `VesselSubType` in the data keep it as `null`.
            - **`certificate_table`:**
                - **Extract All Certificates, Courses, `Flag` Documents, `Visa` Documents, `Passport` and Training:**
                    - Identify and extract every instance of a certificate, course, training session, workshop, `Visa`, `Flag` document, `Passport` and any related document mentioned in the resume. This includes:
                        - **Professional Certificates** (e.g., GMDSS, STCW)
                        - **Courses** (e.g., Maritime, Safety, or Industry-related courses)
                        - **Flag Documents** (e.g., Panama, Marshall Islands flag endorsements)
                        - **Visa Documents** (e.g., US Visa, Schengen Visa)
                        - **Passport** (e.g., Panama Passport)
                        - **Documents** related to mandatory training, workshops, endorsements, and similar entries.
                    - Perform a **deep scan** of the resume to ensure that no relevant certificates, courses, or documents (`Flag` or `Visa` or  `Passport`) are missed, irrespective of their formatting or placement in the content.
                    - **Handle Multi-Line and Scattered Entries**: If a certificate, document, or training entry appears across multiple lines or sections, **merge related information** into a single JSON entry.
                    - Treat all courses, training sessions, workshops, **Flag documents**, **Passport** and **Visa documents** as part of the `certificate_table`.
                    - **Include and Merge Related Certificates and Courses**:
                        - For certificates or documents that are different parts of the same item (e.g., "GMDSS" and "ENDORSEMENT"), combine them into a single JSON entry.
                        - Use a unified `CertificateName` that represents the combined document (e.g., "GMDSS ENDORSEMENT").
                        - Concatenate all related values in other fields where applicable (e.g., merge certificate numbers: "GMDSSCHN22 007367").
                        - Ensure all related fields (`NUMBER`, `ISSUING VALIDATION DATE`, `ISSUING PLACE`) are populated or set to `null` if missing.
                        - Ensure that all `Flag` documents, `Visa` documents, `Passport` and certificates are captured and structured correctly in the final JSON output.
                        - Include all the `Flag` documents, `Visa` documents, `Passport` and certificates even if other details are `null`.

                    - **Handle Special Document Categories (Flag & Visa Documents)**:
                        - Recognize **Flag Documents**: Look for documents that relate to flag endorsements from specific countries (e.g., "Panama Flag", "Marshall Islands Flag") and ensure they are added to the output.
                        - Recognize **Visa Documents**: Identify any visas (e.g., "US Visa", "Schengen Visa") and include them in the output, treating them as documents with relevant fields (such as number, issuing country, and date).
                        - **Merge Related Documents**: If multiple entries refer to the same Flag document or Visa, merge them into a unified entry to avoid duplication.
                    
                    - **Final Validation**:
                        - After extraction, re-scan the resume content to ensure no document or certificate is missing, including all Visa, Passport and Flag-related documents.
                        - If any field (e.g., certificate number, issuing place) is missing, set the value to `null` but still ensure the entry is included in the final output.

                -   Double check and make sure that all certificate entries, including those with missing fields (all `null`), are included and formatted according to the instructions.



            3. **Ensure No Omissions:**
            - Perform a comprehensive scan of the entire resume content to ensure that every certificate, course, training, and document is included in the `certificate_table`.
            - Double-check against the source content to confirm that all entries are captured accurately and completely.
            - Cross-reference with the extracted entries to verify that no certificates, courses, or training are missing from the output.

            4. **Handle Missing Data:**
            - Ensure that any missing or empty fields are set to `null` in the JSON output. Include all rows from the table, even if they contain all `null` values.

            5. **Accuracy and Completeness:**
            - Double-check that all relevant data from the resume content is represented in the output, matching the exact order and structure provided.
            - Include every certificate, course, training, and experience entry, maintaining their original order in the source content.
            - **Ensure Merged Entries are Accurate:** For combined certificates or courses, verify that all related data is included without duplication or loss of information.

            6. **Formatting Standards:**
            - Output should be a well-formatted JSON with proper indentation and without any extra line breaks (`\n`), tabs, or spaces. Each key-value pair should be on its own line with correct indentation.
            - Provide the output directly as JSON, without using code blocks or any additional text.

            7. **Validation and Review:**
            - Review the generated JSON to ensure that all extracted information aligns with the provided template, with no missing entries.
            - Confirm that all sections (`basic_details`, `certificate_table`, `experience_table`) are fully populated according to the template, and that any missing data is set to `null`.
            - **Check All Extracted Entries:** Verify that every certificate, course, training, and document listed in the resume is represented in the output without any omissions.
            - **Check Merged Entries:** Ensure that related certificate entries such as "GMDSS" and "ENDORSEMENT" are merged into a single entry and their fields are properly combined.

            ### **Final Output Requirements:**
            - Provide the output in JSON format only, adhering strictly to the specified template and instructions. Do not include any explanations, comments, or additional text outside of the JSON output.

            8. Don't output anything other than JSON response and also don't use code block.

            9. Generate the output in the exact JSON format and structure specified, making sure that all certificate entries, including those with missing fields (all `null`), are included and formatted according to the instructions.


            """

        response = get_completion(prompt)
        return response

    prompt_1 = generate_prompt(all_predictions, json_template_str)
    json_output = json.loads(prompt_1.content.replace('\n',''))
    print('_______________________________completed_______________________________________')
    return json_output
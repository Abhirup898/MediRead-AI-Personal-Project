import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import easyocr
from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from fuzzywuzzy import fuzz, process
import requests
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class PrescriptionExtractor:
    def __init__(self):
        """Initialize the prescription extraction system"""
        try:
            # Load spaCy model for NER
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Initialize EasyOCR
            self.ocr_reader = easyocr.Reader(['en'])
            
            # Initialize TrOCR for handwritten text (optional, requires GPU for best performance)
            self.trocr_processor = None
            self.trocr_model = None
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                logger.info("TrOCR model loaded successfully")
            except Exception as e:
                logger.warning(f"TrOCR model not available: {e}")
            
            # Common medicine patterns and drug database
            self.medicine_patterns = self._load_medicine_patterns()
            self.common_medicines = self._load_common_medicines()
            
        except Exception as e:
            logger.error(f"Error initializing PrescriptionExtractor: {e}")
            raise

    def _load_medicine_patterns(self) -> List[str]:
        """Load common medicine name patterns"""
        patterns = [
            r'\b\w*cillin\b',  # Penicillin, Amoxicillin, etc.
            r'\b\w*mycin\b',   # Erythromycin, Clindamycin, etc.
            r'\b\w*prazole\b', # Omeprazole, Pantoprazole, etc.
            r'\b\w*olol\b',    # Propranolol, Metoprolol, etc.
            r'\b\w*pine\b',    # Amlodipine, Nifedipine, etc.
            r'\b\w*statin\b',  # Atorvastatin, Simvastatin, etc.
            r'\b\w*pril\b',    # Lisinopril, Enalapril, etc.
            r'\b\w*sartan\b',  # Losartan, Valsartan, etc.
        ]
        return patterns

    def _load_common_medicines(self) -> List[Dict]:
        """Load common medicine database with dosages and frequencies"""
        return [
            {"name": "Paracetamol", "generic": "Acetaminophen", "common_doses": ["500mg", "650mg", "1000mg"]},
            {"name": "Ibuprofen", "generic": "Ibuprofen", "common_doses": ["200mg", "400mg", "600mg"]},
            {"name": "Amoxicillin", "generic": "Amoxicillin", "common_doses": ["250mg", "500mg", "875mg"]},
            {"name": "Omeprazole", "generic": "Omeprazole", "common_doses": ["20mg", "40mg"]},
            {"name": "Metformin", "generic": "Metformin", "common_doses": ["500mg", "850mg", "1000mg"]},
            {"name": "Amlodipine", "generic": "Amlodipine", "common_doses": ["2.5mg", "5mg", "10mg"]},
            {"name": "Lisinopril", "generic": "Lisinopril", "common_doses": ["2.5mg", "5mg", "10mg", "20mg"]},
            {"name": "Atorvastatin", "generic": "Atorvastatin", "common_doses": ["10mg", "20mg", "40mg", "80mg"]},
            {"name": "Aspirin", "generic": "Aspirin", "common_doses": ["81mg", "325mg"]},
            {"name": "Ciprofloxacin", "generic": "Ciprofloxacin", "common_doses": ["250mg", "500mg", "750mg"]},
        ]

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply noise reduction
            image = image.filter(ImageFilter.MedianFilter())
            
            # Convert to OpenCV format for advanced processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            
            # Apply morphological operations to clean text
            kernel = np.ones((1, 1), np.uint8)
            cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            return image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image

    def extract_text_multiple_methods(self, image: Image.Image) -> str:
        """Extract text using multiple OCR methods and combine results"""
        extracted_texts = []
        
        try:
            # Method 1: Tesseract OCR
            tesseract_text = pytesseract.image_to_string(image, config='--psm 6')
            if tesseract_text.strip():
                extracted_texts.append(tesseract_text.strip())
            
            # Method 2: EasyOCR
            if self.ocr_reader:
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                easyocr_results = self.ocr_reader.readtext(cv_image)
                easyocr_text = ' '.join([result[1] for result in easyocr_results])
                if easyocr_text.strip():
                    extracted_texts.append(easyocr_text.strip())
            
            # Method 3: TrOCR for handwritten text
            if self.trocr_processor and self.trocr_model:
                try:
                    pixel_values = self.trocr_processor(images=image, return_tensors="pt").pixel_values
                    generated_ids = self.trocr_model.generate(pixel_values)
                    trocr_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if trocr_text.strip():
                        extracted_texts.append(trocr_text.strip())
                except Exception as e:
                    logger.warning(f"TrOCR extraction failed: {e}")
            
            # Combine and deduplicate results
            if extracted_texts:
                # Use the longest text as primary, supplement with unique words from others
                primary_text = max(extracted_texts, key=len)
                all_words = set()
                for text in extracted_texts:
                    all_words.update(text.split())
                
                # Create comprehensive text
                combined_text = primary_text
                for word in all_words:
                    if word not in combined_text and len(word) > 2:
                        combined_text += f" {word}"
                
                return combined_text
            
            return ""
            
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return ""

    def extract_medicine_entities(self, text: str) -> List[Dict]:
        """Extract medicine names, dosages, and frequencies from text"""
        medicines = []
        
        try:
            # Clean and preprocess text
            text = re.sub(r'\s+', ' ', text.strip())
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Extract potential medicine information from each line
                medicine_info = self._extract_medicine_from_line(line)
                if medicine_info:
                    medicines.append(medicine_info)
            
            # Remove duplicates and validate
            validated_medicines = self._validate_and_clean_medicines(medicines)
            
            return validated_medicines
            
        except Exception as e:
            logger.error(f"Error in medicine entity extraction: {e}")
            return []

    def _extract_medicine_from_line(self, line: str) -> Optional[Dict]:
        """Extract medicine information from a single line"""
        try:
            # Patterns for different prescription formats
            dosage_pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?|iu)'
            frequency_patterns = [
                r'\b(once|twice|thrice|\d+\s*times?)\s*(daily|day|a\s*day)\b',
                r'\b(od|bd|td|qd|bid|tid|qid)\b',
                r'\b(\d+)\s*x\s*(\d+)\b',
                r'\b(morning|evening|night|bedtime)\b'
            ]
            
            # Find potential medicine name (usually at the beginning)
            words = line.split()
            medicine_name = None
            
            # Look for medicine name using fuzzy matching with known medicines
            for i, word in enumerate(words):
                if len(word) > 3:  # Skip very short words
                    best_match = self._find_best_medicine_match(word)
                    if best_match:
                        medicine_name = best_match
                        break
                    
                    # Check if word matches medicine patterns
                    for pattern in self.medicine_patterns:
                        if re.search(pattern, word, re.IGNORECASE):
                            medicine_name = word
                            break
                    
                    if medicine_name:
                        break
            
            # If no exact match, use the first substantial word as potential medicine
            if not medicine_name and words:
                substantial_words = [w for w in words if len(w) > 3 and not re.match(r'^\d+', w)]
                if substantial_words:
                    medicine_name = substantial_words[0]
            
            if not medicine_name:
                return None
            
            # Extract dosage
            dosage_match = re.search(dosage_pattern, line, re.IGNORECASE)
            dosage = f"{dosage_match.group(1)}{dosage_match.group(2)}" if dosage_match else None
            
            # Extract frequency
            frequency = None
            for pattern in frequency_patterns:
                freq_match = re.search(pattern, line, re.IGNORECASE)
                if freq_match:
                    frequency = freq_match.group(0)
                    break
            
            # Extract additional instructions
            instructions = self._extract_instructions(line)
            
            return {
                "medicine_name": medicine_name.title(),
                "dosage": dosage,
                "frequency": frequency,
                "instructions": instructions,
                "original_text": line,
                "confidence": self._calculate_confidence(medicine_name, dosage, frequency)
            }
            
        except Exception as e:
            logger.error(f"Error extracting medicine from line '{line}': {e}")
            return None

    def _find_best_medicine_match(self, word: str) -> Optional[str]:
        """Find best matching medicine name using fuzzy matching"""
        try:
            medicine_names = [med["name"] for med in self.common_medicines]
            best_match = process.extractOne(word, medicine_names, scorer=fuzz.ratio)
            
            if best_match and best_match[1] > 70:  # 70% similarity threshold
                return best_match[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {e}")
            return None

    def _extract_instructions(self, line: str) -> str:
        """Extract additional instructions from the prescription line"""
        instruction_keywords = [
            'before meals', 'after meals', 'with food', 'empty stomach',
            'as needed', 'for pain', 'for fever', 'continue for', 'days',
            'weeks', 'months', 'review', 'follow up'
        ]
        
        instructions = []
        line_lower = line.lower()
        
        for keyword in instruction_keywords:
            if keyword in line_lower:
                # Extract surrounding context
                start_idx = line_lower.find(keyword)
                context_start = max(0, start_idx - 10)
                context_end = min(len(line), start_idx + len(keyword) + 15)
                context = line[context_start:context_end].strip()
                instructions.append(context)
        
        return '; '.join(instructions) if instructions else ""

    def _calculate_confidence(self, medicine_name: str, dosage: str, frequency: str) -> float:
        """Calculate confidence score for extracted medicine information"""
        confidence = 0.0
        
        # Base confidence for having a medicine name
        if medicine_name:
            confidence += 0.4
            
            # Bonus for recognized medicine
            if self._find_best_medicine_match(medicine_name):
                confidence += 0.3
        
        # Bonus for having dosage
        if dosage:
            confidence += 0.2
        
        # Bonus for having frequency
        if frequency:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _validate_and_clean_medicines(self, medicines: List[Dict]) -> List[Dict]:
        """Validate and clean extracted medicines"""
        validated = []
        seen_medicines = set()
        
        for medicine in medicines:
            medicine_name = medicine.get("medicine_name", "").lower()
            
            # Skip duplicates
            if medicine_name in seen_medicines:
                continue
            
            # Skip if medicine name is too short or generic
            if len(medicine_name) < 3 or medicine_name in ['tab', 'cap', 'syp', 'inj']:
                continue
            
            # Add validation against known medicine database
            validated_medicine = self._validate_against_database(medicine)
            if validated_medicine:
                validated.append(validated_medicine)
                seen_medicines.add(medicine_name)
        
        return validated

    def _validate_against_database(self, medicine: Dict) -> Optional[Dict]:
        """Validate medicine against known database and enrich information"""
        try:
            medicine_name = medicine.get("medicine_name", "")
            
            # Find matching medicine in database
            for known_med in self.common_medicines:
                if (fuzz.ratio(medicine_name.lower(), known_med["name"].lower()) > 70 or
                    fuzz.ratio(medicine_name.lower(), known_med["generic"].lower()) > 70):
                    
                    # Enrich with database information
                    medicine.update({
                        "generic_name": known_med["generic"],
                        "brand_name": known_med["name"],
                        "common_doses": known_med["common_doses"],
                        "validated": True
                    })
                    return medicine
            
            # If not found in database, still return with lower confidence
            medicine["validated"] = False
            return medicine
            
        except Exception as e:
            logger.error(f"Error validating medicine: {e}")
            return medicine

    def process_prescription(self, image_data: str) -> Dict:
        """Main method to process prescription image and extract medicine information"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text
            extracted_text = self.extract_text_multiple_methods(processed_image)
            
            if not extracted_text:
                return {
                    "success": False,
                    "error": "No text could be extracted from the image",
                    "medicines": []
                }
            
            # Extract medicine information
            medicines = self.extract_medicine_entities(extracted_text)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "medicines": medicines,
                "total_medicines": len(medicines),
                "processed_at": datetime.now().isoformat(),
                "confidence_score": sum(med.get("confidence", 0) for med in medicines) / len(medicines) if medicines else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing prescription: {e}")
            return {
                "success": False,
                "error": str(e),
                "medicines": []
            }

# Initialize the extractor
extractor = PrescriptionExtractor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "MediRead AI Backend"
    })

@app.route('/extract', methods=['POST'])
def extract_prescription():
    """Main endpoint to extract medicine information from prescription image"""
    try:
        # Check if request contains image data
        if not request.json or 'image' not in request.json:
            return jsonify({
                "success": False,
                "error": "No image data provided. Please send base64 encoded image in 'image' field."
            }), 400
        
        image_data = request.json['image']
        
        # Process the prescription
        result = extractor.process_prescription(image_data)
        
        # Return appropriate status code
        status_code = 200 if result["success"] else 400
        
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in extract_prescription endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error occurred while processing the prescription",
            "details": str(e)
        }), 500

@app.route('/validate_medicine', methods=['POST'])
def validate_medicine():
    """Endpoint to validate a medicine name against the database"""
    try:
        if not request.json or 'medicine_name' not in request.json:
            return jsonify({
                "success": False,
                "error": "No medicine name provided"
            }), 400
        
        medicine_name = request.json['medicine_name']
        
        # Find best match
        best_match = extractor._find_best_medicine_match(medicine_name)
        
        if best_match:
            # Find full information
            for med in extractor.common_medicines:
                if med["name"] == best_match:
                    return jsonify({
                        "success": True,
                        "validated": True,
                        "medicine": med,
                        "similarity_score": fuzz.ratio(medicine_name.lower(), best_match.lower())
                    })
        
        return jsonify({
            "success": True,
            "validated": False,
            "message": f"Medicine '{medicine_name}' not found in database"
        })
        
    except Exception as e:
        logger.error(f"Error in validate_medicine endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/medicines', methods=['GET'])
def get_medicines_database():
    """Endpoint to get the list of known medicines"""
    try:
        return jsonify({
            "success": True,
            "medicines": extractor.common_medicines,
            "total_count": len(extractor.common_medicines)
        })
        
    except Exception as e:
        logger.error(f"Error in get_medicines_database endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
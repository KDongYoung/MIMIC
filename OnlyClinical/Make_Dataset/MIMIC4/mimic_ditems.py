import re
import gzip
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

def ditems(icu_path):
    print(f"Preprocess d_items.csv")
    
    d_items = pd.read_csv(gzip.open(f'{icu_path}/d_items.csv.gz'))
    
    d_items.loc[:, 'category']=d_items.loc[:, 'category'].str.replace('Scores - APACHE IV (2)', 'Scores - APACHE IV', regex=False)

    d_items=d_items[(d_items["category"]!='Alarms') &
                    (d_items['category'] != "Scores - APACHE II") & 
                    (d_items['category'] != "Scores - APACHE IV") & 
                    (d_items['category'] != "ApacheII Parameters") & 
                    (d_items['category'] != "ApacheIV Parameters")]

    d_items=d_items[(d_items["category"] == 'Blood Products-Colloids') |
                    (d_items['category'] == "Fluids-Intake") | 
                    (d_items['category'] == "Medications") | 
                    (d_items['category'] == "Nutrition-Enteral") |
                    (d_items['category'] == "Nutrition-Supplements") |
                    (d_items['category'] == "Nutrition-Parenteral") |
                    (d_items['category'] == "Antibiotics") |
                    (d_items['category'] == "Drains") |
                    (d_items['category'] == "Output") | 
                    (d_items['category'] == "General") |
                    (d_items["category"] == 'Routine Vital Signs') |
                    (d_items['category'] == "Hemodynamics") | 
                    (d_items['category'] == "Respiratory") | 
                    (d_items['category'] == "Neurological") | 
                    (d_items['category'] == "Pain-Sedation") |
                    (d_items["category"] == 'Toxicology') |
                    (d_items['category'] == "Cardiovascular") |
                    (d_items["category"] == 'Pulmonary') |
                    (d_items['category'] == "Treatments") |
                    (d_items['category'] == "Labs")]

    ##### Re name the same labels (d_items and d_labitems)
    label=[['Absolute Count Basos', 'Absolute Basophil Count'], ['Absolute Count Eos', 'Absolute Eosinophil Count'],
            ['Absolute Count Lymphs', 'Absolute Lymphocyte Count'], ['Absolute Count Monos', 'Absolute Monocyte Count'],
            ['Absolute Count Neuts', 'Absolute Neutrophil Count'], ['Absolute Neutrophil Count', 'Absolute Count Neuts'],
            # ['Albumin', 'Albumin'], ['Alkaline Phosphate', 'Alkaline Phosphatase'], 
            ['ALT', 'Alanine Aminotransferase (ALT)'],
            # ['Ammonia', 'Ammonia'], ['Amylase', 'Amylase'],
            ['Anion gap', 'Anion Gap'], ['AST', 'Asparate Aminotransferase (AST)'],
            ['C Reactive Protein (CRP)', 'C-Reactive Protein'], ['Chloride (serum)', 'Chloride'],
            ['Chloride (whole blood)', 'Chloride, Whole Blood'], ['Cholesterol', 'Cholesterol, Total'],
            ['HDL', 'Cholesterol, HDL'], ['LDL calculated', 'Cholesterol, LDL, Calculated'],
            ['LDL measured', 'Cholesterol, LDL, Measured'], ['Creatinine (serum)', 'Creatinine'],
            ['Creatinine (whole blood)', 'Creatinine, Whole Blood'], ['Fibrinogen', 'Fibrinogen, Functional'],
            # ['D-Dimer', 'D-Dimer'], ['Digoxin', 'Digoxin'], 
            ['Gentamicin (Peak)', 'Gentamicin'], ['Gentamicin (Random)', 'Gentamicin'],
            ['Gentamicin (Trough)', 'Gentamicin'], ['Glucose (serum)', 'Glucose'],
            ['Glucose (whole blood)', 'Glucose, Whole Blood'], ['Hematocrit (serum)', 'Hematocrit'],
            ['Hemoglobin', 'Hemoglobin'], ['INR', 'INR(PT)'],
            # ['Lipase', 'Lipase'], ['Magnesium', 'Magnesium'],
            ['PH (Arterial)', 'pH'], ['PH (dipstick)', 'pH'],
            ['PH (Venous)', 'pH'], ['Platelet Count', 'Platelet Count'],
            ['Phenytoin (Dilantin)', 'Phenytoin'], ['Phenytoin (Free)', 'Phenytoin, Free'],
            # ['PO2 (Mixed Venous)', 'pO2'], ['Phenobarbital', 'Phenobarbital'],
            ['Potassium (serum)', 'Potassium'], ['Potassium (whole blood)', 'Potassium, Whole Blood'],
            # ['PTT', 'PTT'],  ['Uric Acid', 'Uric Acid'],        
            ['Sodium (whole blood)', 'Sodium, Whole Blood'], ['Specific Gravity (urine)', 'Specific Gravity'],
            ['Thrombin', 'Thrombin'], ['Tobramycin (Peak)', 'Tobramycin'],
            ['Tobramycin (Random)', 'Tobramycin'], ['Tobramycin (Trough)', 'Tobramycin'],
            ['Direct Bilirubin', 'Bilirubin, Direct'], ['Total Bilirubin', 'Bilirubin, Total'],
            ['Total Granulocyte Count (TGC)', 'Granulocyte Count'], ['Triglyceride', 'Triglycerides'],
            ['TroponinT', 'Troponin T'], ['Sodium (serum)', 'Sodium'],
            ['Vancomycin (Peak)', 'Vancomycin'], ['Vancomycin (Random)', 'Vancomycin'],
            ['Vancomycin (Trough)', 'Vancomycin'],
            ['WBC','ICU_WBC'] # labevents의 wbc와 값이 다름,,, why?!
        ]
    for original, replacement in label:
        d_items['label'] = d_items['label'].str.replace(original, replacement, regex=False)

    ## Function to replace `#number` with a specific string
    def replace_number(s, replacement):
        return re.sub(r' # ?\d*', replacement, s)

    # 특정 패턴 제거 함수
    def remove_patterns(text):
        text = text.replace('  -', '')
        text = text.replace(' -', '')
        text = text.replace('-', '')
        return text

    replacement = ""
    d_items['unique_label'] = [replace_number(s, replacement) for s in d_items["label"]]
    contain_label = list(set([replace_number(s, replacement) for s in d_items['label'] if re.search(r'#', s)]))
    d_items['unique_label'] = d_items['unique_label'].apply(remove_patterns)

    ## Label에 ‘alarm’이 포함된 것 제거
    def clean_text(text):
        alarms_to_remove = [
            'High Power Alarm (HeartWare)', 
            'Low Flow Alarm (HeartWare)', 
            'High Watts Alarm (VAD)', 
            'Low Flow Alarm (VAD)',
            'Alarm (Hi)',
            'Alarm (Lo)',
            'systolic',
            'diastolic',
            'Systolic',
            'Diastolic'
            ]
        
        if any(alarm in text for alarm in alarms_to_remove):
            return None
        return text

    d_items.loc[:, 'label'] = d_items['label'].apply(clean_text)
    d_items.dropna(subset=['label'], inplace=True)
    
    categorical = {row['unique_label']: "categorical" for _, row in d_items[d_items['unitname'].isnull()].iterrows()} # 과연 단위가 없는 것을 모두 CATEGORICLA로 봐도 될까...
    numerical = {row['unique_label']: "numerical" for _, row in d_items[d_items['unitname'].notnull()].iterrows()}
    column_dictionary = {**categorical, **numerical} ## combine two dictionary
    
    return d_items, column_dictionary, contain_label
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services import file_service as fs
from services import article_service as as_
from services import text_analysis_service as tas
import json
from secrets import token_hex
import os
import pandas as pd
import base64

app = FastAPI()

origins = [
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_path = "uploaded/"

@app.get("/")
async def hello():
    return {"message": "Hello from Papers"}


# file processing endpoint
@app.post("/api/process-files")
async def process_files(seed_file: UploadFile = File(...), article_file: UploadFile = File(...)):
    try:
        seed_file_name = seed_file.filename
        article_file_name = article_file.filename
        
        # validate file names
        warnings = fs.validate_file_names(seed_file_name, article_file_name)

        if warnings:
            raise HTTPException(status_code=400, detail=json.dumps({"warnings": warnings}))

        seed_file_path = await fs.save_uploaded_file(seed_file)
        article_file_path = await fs.save_uploaded_file(article_file)

        # files reading
        seed_df = await fs.read_file(seed_file_path)
        article_df = await fs.read_file(article_file_path)

        # verify "title" and "abstract" cols
        fs.check_columns(seed_df, "seeds")
        fs.check_columns(article_df, "articles")

        # add seed column
        seed_df = fs.add_is_seed_column(seed_df)
        article_df = fs.add_is_seed_column(article_df)
        article_df["is_seed"] = 0

        # calculate seeds, duplicates and unique
        seed_elements, seed_duplicates, seed_unique = fs.calculate_duplicates(seed_df, ["title", "abstract"])
        article_elements, article_duplicates, article_unique = fs.calculate_duplicates(article_df, ["title", "abstract"])

        # Convertir les valeurs entières en int
        seed_elements = int(seed_elements)
        seed_duplicates = int(seed_duplicates)
        seed_unique = int(seed_unique)
        article_elements = int(article_elements)
        article_duplicates = int(article_duplicates)
        article_unique = int(article_unique)

        # process the two files to conduct the review
        merged_df = pd.concat([seed_df, article_df], ignore_index=True)
        # add a label_col
        merged_df['label_included'] = merged_df['is_seed']

        # create a TextPreprocessor instance
        preprocessor = as_.TextPreprocessor(title_col='title', abstract_col='abstract', title_abstract_col='title_abstract')

        merged_df_clean = preprocessor.analyze_data(df=merged_df, label_col="label_included")

        # create a SimilarityService instance
        textAnalysis = tas.TextAnalysisService()
        tf_idf_matrix, merged_df_clean = textAnalysis.calculate_tf_idf_matrix(merged_df_clean, text_column='title_abstract')

        # get the number of seeds
        seeds = seed_df.shape[0]

        # analyse similarities
        merged_df_clean_similarities = textAnalysis.analyze_similarity(
            dataset="Papa", seeds=seeds, tf_idf_matrix=tf_idf_matrix,
            df_cleaned_sorted=merged_df_clean, relevant_docs=seed_df.shape[0], total_docs=merged_df_clean.shape[0]
        )
        # convert similarities to float
        merged_df_clean_similarities['similarity'] = merged_df_clean_similarities['similarity'].astype(float)

        # filter columns
        merged_df_clean_similarities = merged_df_clean_similarities[["title", "similarity", "label_included"]]

        # Arrondir les valeurs flottantes à deux décimales
        merged_df_clean_similarities['similarity'] = merged_df_clean_similarities['similarity'].round(3)

        # add the probability col
        merged_df_clean_similarities = tas.transform_to_probabilities(merged_df_clean_similarities, number_column='similarity', probability_col_name='probability')

        csv_filename = (seed_file.filename.split('_')[0] + '_results.csv').lower()

        csv_path = os.path.join("outputs", csv_filename)

        merged_df_clean_similarities.loc[merged_df_clean_similarities['similarity']>0,"label_included"] = 1
        merged_df_clean_similarities[seeds:].to_csv(csv_path, index=False)

        # transform datto binary
        with open(csv_path, 'rb') as f:
            csv_data = f.read()

        encoded_csv_data = base64.b64encode(csv_data).decode('utf-8')

        # penser à vérifier que les calculs se font entre les seeds et les non seeds
        return {
            "success": True,
            "message": "Fichier csv créé avec succès",
            "seed_file": {
            #    "head": seed_df.head().to_dict(),
                "total_elements": seed_elements,
                "duplicates": seed_duplicates,
                "unique_elements": seed_unique
            },
            "article_file": {
            #    "head": article_df.head().to_dict(),
                "total_elements": article_elements,
                "duplicates": article_duplicates,
                "unique_elements": article_unique
            },
            "csv_content": encoded_csv_data
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite lors du traitement des fichiers : {str(e)}")

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app', host="127.0.0.1", reload=True)
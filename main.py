import json
from typing import List, Union, Optional

from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# import uvicorn
import pandas as pd
import joblib
from scripts.target_scripts import *
import warnings

from scripts.weather_scripts import prepare_weather_data

warnings.filterwarnings("ignore")
app = FastAPI()

app.mount("/interface/", StaticFiles(directory="frontend", html=True))
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def upload_files(files: List[UploadFile], directory: str):
    if len(files) == 0:
        raise HTTPException(status_code=404, detail="No files uploaded")

    for file in files:
        try:
            file_location = os.path.join(directory, file.filename)
            with open(file_location, "wb") as f:
                f.write(file.file.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail="Files uploading failed") from e


@app.post("/datasets/prepare")
async def prepare_datasets():
    try:
        create_datasets()
        return "Dataset prepared successfully", 200
    except Exception as e:
        print(e)
        return "Dataset preparation failed", 500


@app.post("/datasets/upload_predict")
async def upload_predict_files(files: List[UploadFile]):
    """
    Parameters:
    directory - папка куда выгружаем необходимые файлы
    """
    directory = "uploaded"
    try:
        upload_files(files, directory)

    except Exception as e:
        print(e)
        return JSONResponse(
            content={"error": "Files uploading failed"}, status_code=500
        )

    try:
        prepare_datasets()
    except Exception as e:
        print(e)
        return JSONResponse(
            content={"error": "Files preparing failed"}, status_code=500
        )
    return JSONResponse(content={"message": "Success"}, status_code=200)


@app.post("/datasets/weather_upload")
async def upload_weather_file(files: List[UploadFile] = File(...)):
    """
    Parameters:
    directory - папка куда выгружаем необходимые файлы
    """
    directory = "weather_input"
    try:
        upload_files(files, directory)
    except HTTPException as e:
        return JSONResponse(content={"error": str(e.detail)}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse(
            content={"error": "Files uploading failed"}, status_code=500
        )

    try:
        prepare_weather_data("weather_input/", "weather_gzip_input/", "weather_dfs/")
    except Exception as e:
        print(e)
        return JSONResponse(
            content={"error": "Files preparing failed"}, status_code=500
        )

    return JSONResponse(content={"message": "Success"}, status_code=200)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/model/predict")
def predict(
    ks: str = Query(...),
    pmode: str = Query(...),
    hours: str = Query(...),
    date: str = Query(...),
):
    """
    Parameters:
    KC - номер КС (15, 16, 17, 19 и тд)
    Pmode - Pin/Pout
    Hours - 48h, 72h, 96h
    date - дата, дд.мм.гггг
    """
    try:
        # Load the model
        model_path = f"ModelsNewFix/{pmode}_lag_{hours}_КС-{ks}_LGBM_model.joblib"
        loaded_model = joblib.load(model_path)

        # Load training and test data
        train_path = f"data/КС-{ks}_{pmode}_{hours[:-1]}_h.parquet"
        test_path = f"data/КС-{ks}_{pmode}_{hours[:-1]}_h_new.parquet"

        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)

        # Filter test data by date
        test = test[test["DateTime"].dt.strftime("%d.%m.%Y") == date]
        if len(test) == 0:
            test = train[train["DateTime"].dt.strftime("%d.%m.%Y") == date]
            if len(test) == 0:
                raise HTTPException(
                    status_code=404, detail="Data not found for the given date"
                )

        # Prepare training data
        X_train = train.drop(columns=["DateTime", f"{pmode}_target_shift_{hours}_{ks}"])
        y_train = train[f"{pmode}_target_shift_{hours}_{ks}"]

        # Fit the model
        loaded_model.fit(X_train, y_train)

        # Prepare test data
        X_test = test.drop(columns=["DateTime"])

        # Create date index for the predictions
        date_index = test["DateTime"].apply(
            lambda x: x + pd.Timedelta(hours=int(hours[:-1]))
        )
        date_index = pd.to_datetime(date_index, errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M"
        )

        # Make predictions
        result = loaded_model.predict(X_test)
        prediction = pd.concat(
            [
                date_index.reset_index(drop=True),
                pd.DataFrame(result).apply(lambda x: round(x, 3)),
            ],
            axis=1,
        ).rename(columns={0: "Result"})

        # Add additional fields
        prediction["mode"] = pmode
        prediction["station"] = ks

        # Save prediction to Excel
        result_path = os.path.join(
            "result", f"result_КС-{ks}_{pmode}__{hours[:-1]}_h.xlsx"
        )
        prediction.to_excel(result_path, index=False)

        return JSONResponse(content=json.loads(prediction.to_json(orient="records")))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def read_item(
    q: Union[str, None] = None,
    ks: Union[str, None] = None,
    pmode: Union[str, None] = None,
    hours: Union[str, None] = None,
    date: Union[str, None] = None,
):
    return {"result": [q, ks, pmode, hours, date]}


@app.get("/results/download")
def download_file(
    ks: str = Query(...), pmode: str = Query(...), hours: str = Query(...)
):
    """
    Parameters:
    KC - номер КС (15, 16, 17, 19 и тд)
    Pmode - Pin/Pout
    Hours - 48h, 72h, 96h
    date - дата, дд.мм.гггг
    """
    # Construct the file path based on query parameters
    file_path = os.path.join("result/", f"result_КС-{ks}_{pmode}__{hours[:-1]}_h.xlsx")

    # Try to send the file, handle errors appropriately
    try:
        return FileResponse(
            file_path,
            # media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"result_КС-{ks}_{pmode}__{hours[:-1]}_h.xlsx",
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Files sending failed")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

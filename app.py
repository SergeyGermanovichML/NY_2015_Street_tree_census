import joblib
import torch
from fastapi import FastAPI
import uvicorn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from pydantic import BaseModel

app = FastAPI()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Определение модели
class SimpleMLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def load_model(path: str):
    """Function for load model"""
    model = torch.load(path, map_location=device)
    model.to(device)
    return model


class TreeFeatures(BaseModel):
    tree_dbh: float
    curb_loc_OnCurb: int
    steward_3or4: int
    steward_4orMore: int
    steward_No: int
    guards_Helpful: int
    guards_No: int
    guards_Unsure: int
    sidewalk_NoDamage: int
    root_stone_Yes: int
    root_grate_Yes: int
    trunk_wire_Yes: int
    trnk_light_Yes: int
    brch_light_Yes: int
    brch_other_Yes: int
    borough_Brooklyn: int
    borough_Manhattan: int
    borough_Queens: int
    borough_Staten_Island: int

# Определяем модель и переводим режим инференса
model = load_model("NY_2015_model.pth")
model.eval()
# Загружаем scaler
scaler = joblib.load("scaler.pkl")


@app.get("/get_health_condition/")
def get_health_condition(features: TreeFeatures) -> str:
    """Function that take features and return health condition"""
    features.tree_dbh = float(scaler.transform(np.array(features.tree_dbh).reshape(-1, 1)))
    input_data = np.array(list(features.model_dump().values()))
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).numpy()
        prediction = np.argmax(probabilities)

    health_classes = {2: "Good", 1: "Fair", 0: "Poor"}
    return {"predicted_health": health_classes[prediction]}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost')

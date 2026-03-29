"""Pydantic schemas for validated property inputs."""

from typing import Optional

from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    """Single listing features aligned with training columns in `src/preprocess.py`."""

    suburb: str = Field(..., description="Melbourne suburb name")
    property_type: str = Field(
        ...,
        alias="Type",
        description="Property type (h, u, t, etc.)",
    )
    regionname: str = Field(..., description="Region name (e.g. Northern Metropolitan)")
    rooms: int = Field(..., ge=1, le=20)
    distance: float = Field(..., ge=0, description="km from CBD")
    postcode: float = Field(..., ge=3000, le=4000)
    bedroom2: float = Field(..., ge=0, le=20)
    bathroom: float = Field(..., ge=0, le=15)
    car: float = Field(..., ge=0, le=20)
    landsize: float = Field(..., ge=0)
    building_area: Optional[float] = Field(None, alias="BuildingArea", ge=0)
    year_built: Optional[float] = Field(None, alias="YearBuilt", ge=1800, le=2030)
    lattitude: float = Field(..., alias="Lattitude", ge=-90, le=0)
    longtitude: float = Field(..., alias="Longtitude", ge=0, le=180)
    propertycount: float = Field(..., ge=0, alias="Propertycount")

    model_config = {"populate_by_name": True}

    def to_feature_frame(self):
        """Build a one-row pandas DataFrame with training column names."""
        import pandas as pd

        row = {
            "Suburb": self.suburb,
            "Type": self.property_type,
            "Regionname": self.regionname,
            "Rooms": self.rooms,
            "Distance": self.distance,
            "Postcode": self.postcode,
            "Bedroom2": self.bedroom2,
            "Bathroom": self.bathroom,
            "Car": self.car,
            "Landsize": self.landsize,
            "BuildingArea": self.building_area,
            "YearBuilt": self.year_built,
            "Lattitude": self.lattitude,
            "Longtitude": self.longtitude,
            "Propertycount": self.propertycount,
        }
        return pd.DataFrame([row])

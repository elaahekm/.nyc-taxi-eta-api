import geopandas as gpd

zones = gpd.read_file("data/taxi_zones/taxi_zones/taxi_zones.shp")
print("original CRS:", zones.crs)

# some zones have several polygons, merge them into one row per zone
zones = zones.dissolve(by="LocationID")

# compute centers in the projected system (accurate), then convert to lat/lon
centers = zones.geometry.centroid.to_crs(epsg=4326)

out = centers.to_frame("geometry")
out["lat"] = centers.y
out["lon"] = centers.x
out = out[["lat", "lon"]].reset_index()

print(out.shape)
print(out.head())

out.to_csv("data/zone_centers.csv", index=False)
print("saved data/zone_centers.csv")
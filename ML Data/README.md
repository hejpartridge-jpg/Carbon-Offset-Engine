# GLanCE: A Global Land Cover Training Dataset from 1984 to 2020
The Global Land Cover Estimation (GLanCE) project provides high-quality long-term records of land cover and land cover change at 30 m spatial resolution for the 21st century using the Landsat archive. The GLanCE land cover classification scheme is designed to be compatible with common land cover classification systems such as the IPCC land use categories for greenhouse gas inventory reporting and the FAO Land Cover Classification System (LCCS). As part of the GLanCE project, we present a new land cover training database that is designed to provide global coverage, ensure accuracy of land cover labels at 30 m spatial resolution, cover nearly four decades, and produce a geographically dense dataset.

## Download Dataset
This repository contains a [Cloud-Native Geospatial](https://cloudnativegeo.org/) [GeoParquet](https://geoparquet.org/) version as well as a [GeoJSON](https://geojson.org/) version of the data. You can use the [BROWSE link](https://beta.source.coop/boston-university/bu-glance) on the left hand side to navigate to the data and download it. You can also access the S3 bucket directly with S3 tools (aws cli, boto3, etc); the S3 URI is s3://us-west-2.opendata.source.coop/boston-university/bu-glance/


## Creator & Contact
* [Boston University Global Land Cover Estimation (GLanCE)](https://sites.bu.edu/measures/)
* rkstan@bu.edu 

## Data License
The data is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). 

## Citation & DOI
Stanimirova R., Tarrio K., Turlej K., McAvoy K., Stonebrook S., Hu K-T., Arévalo P., Bullock E.L., Zhang Y., Woodcock C.E., Olofsson P., Zhu Z., Barber C.P., Souza C., Chen S., Wang J.A., Mensah F., Calderón-Loor M., Hadjikakou M., Bryan B.A., Graesser J., Beyene D.L., Mutasha B., Siame S., Siampale A., and M.A. Friedl (2023) "A Global Land Cover Training Dataset from 1984 to 2020", Version 1.0, Radiant MLHub. [Date Accessed] https://doi.org/10.34911/rdnt.x4xfh3

## Publication 
Stanimirova, R., Tarrio, K., Turlej, K., McAvoy K., Stonebrook S., Hu K-T., Arévalo P., Bullock E.L., Zhang Y., Woodcock C.E., Olofsson P., Zhu Z., Barber C.P., Souza C., Chen S., Wang J.A., Mensah F., Calderón-Loor M., Hadjikakou M., Bryan B.A., Graesser J., Beyene D.L., Mutasha B., Siame S., Siampale A., and M.A. Friedl (2023) A global land cover training dataset from 1984 to 2020. Sci Data 10, 879 https://doi.org/10.1038/s41597-023-02798-5


| Column Name | Description |
| ------------- | ------------- |
| Lat  | Latitude  |
| Lon  | Longitude  |
| Start_Year  | Start year of segment, ranging from 1984 to 2020 (integer) |
| End_Year  | End year of segment, ranging from 1984 to 2020 (integer) |
| Glance_Class_ID_level1  | Level 1 land cover value (integer): 1 (Water), 2 (Ice/snow), 3 (Developed), 4 (Barren/sparsely vegetated), 5 (Trees), 6 (Shrub), and 7 (Herbaceous). |
| Glance_Class_ID_level2  | Level 2 land cover value (integer): 1 (Water), 2 (Ice/snow), 3 (Developed), 4 (Soil), 5 (Rock), 6 (Beach/sand), 7 (Deciduous), 8 (Evergreen), 9 (Mixed), 10 (Shrub), 11 (Grassland), 12 (Agriculture), and 13 (Moss/lichen). NaN values present. |
| Leaf_Type  | Tree leaf type: broadleaf (1), needleleaf (2), and mixed (3). NaN values present.  |
| Impervious_Percent  | Impervious percent for developed samples: low 0%-30% (1), medium 30%-60% (2), and high 60%-100% (3). NaN values present.  |
| Tree_Location  | Binary integer indicating whether trees are on the interior (0) or edge (1) of a forest. NaN values present.   |
| Veg_Density  | Vegetation density for trees and shrubs: sparse 0%-30% (1), open 30%-60% (2), and closed 60%-100% (3). NaN values present.   |
| Veg_Modifier | Vegetation modifiers, which can include one or more of the following: Cropland, Plantation, Wetland, Riparian/Flood, Mangrove, Greenhouse, and Trees/Shrub Present. NaN values present.    |
| Segment_Type  | Indicates whether a segment is stable (0) or transitional (1). See Section 1 for a detailed description. Land cover for transitional segments is recorded at both the beginning and end of the time segment - typically the first and last three years. NaN values present.    |
| Change  | Indicates presence (1) or absence (0) of land cover change for Level 1 land cover labels. Includes both abrupt change and gradual change (transitional segments (1) from the Segment_Type attribute) if it happened at any time for that training unit.   |
| LC_Confidence  | Interpreter confidence in the Level 1 land cover label from 1 (lowest) to 3 (highest). NaN values present.  |
| Level1_Ecoregion  | Ecoregion Level 1 number based on World Wildlife Fund definitions. For North America we used ecoregions based on the Environmental Protection Agency’s Ecoregions of North America product. |
| Level2_Ecoregion  | Ecoregion Level 2 number based on the Environmental Protection Agency’s Ecoregions of North America product. This field is available only for North America and is assigned a value of 0 for all other continents.   |
| Continent_Code  | Assigned continent number: North America (1), South America (2), Africa (3), Europe (4), Asia (5), and Oceania (6). |
| Dataset_Code |  Assigned dataset number: 1, 2, 3, 4, 5, 902, 999, 700, 701, 702, 703, 704, 705, 706, and 707. Numbers correspond to each Dataset as follows: STEP, CLUSTERING, LCMAP, ABoVE, MapBiomas, Feedback, Training_augment, MODIS_algo, GeoWiki, RadEarth, Collaborator_data, BU_team_collected, GLC30, LUCAS, ASB_crop.  For details see Scientific Data publication.    |
| Glance_ID  |  Unique ID for each sample.  |
| ID  |  ID for each unique combination of latitude and longitude. Change units have the same ID but different Glance_ID.   |

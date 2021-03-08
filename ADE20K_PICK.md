# **清理ADE20K数据用于训练室内分割**


## **数据来源**
- ADE20K
- PASCAL_Context
- PASCAL_VOC_2012
- COCOPanoptic
- SUNRGBD

## **Jun.Chen  17 class**

| 序号  |                 类别名                 |  中文名  | 像素占比 |
| :---: | :------------------------------------: | :------: | :------: |
|   1   |               background               |   背景   |   0.6    |
|   2   |                 floor                  |   地板   |   0.1    |
|   3   |                  bed                   |    床    | 像素占比 |
|   4   |    cabinet,wardrobe,bookcase,shelf     |  架子类  | 像素占比 |
|   5   |                 person                 |   行人   | 像素占比 |
|   6   |                  door                  |    门    | 像素占比 |
|   7   |           table,desk,coffee            |  桌子类  | 像素占比 |
|   8   | chair,armchair,sofa,bench,swivel,stool |  椅子类  | 像素占比 |
|   9   |                  rug                   |   地毯   | 像素占比 |
|  10   |                railing                 |   栏杆   | 像素占比 |
|  11   |                 column                 |   柱子   | 像素占比 |
|  12   |              refrigerator              |   冰箱   | 像素占比 |
|  13   |          stairs,stairway,step          |  楼梯类  | 像素占比 |
|  14   |               escalator                | 自动扶梯 | 像素占比 |
|  15   |                  wall                  |   墙壁   | 像素占比 |
|  16   |                  dog                   |    狗    | 像素占比 |
|  17   |                 plant                  |   盆栽   | 像素占比 |

---

## **Tips**

---
1. idx为137，49的图像不作为训练
2. idx为2的设置为背景
3. 和植物链接在一起的瓶子要合并和植物(暂时的思路)

## **Yaoshun.Li ade20k 22 class**
1. 墙面,窗帘、绘画  1、19、23，59,64,101,102,124,131,147
2. 天花板     6
3. 椅子，柜子类、桌子   11，16，20，25，31，32，34，35,36，45，57,63,65,70,71,72,74，76，89，98,111,113,119,122,125,130
4. 植物，树 18，5,67,73
5. 沙发 24、31
6. 地面 4, 12，30,55
7. 床 8, 118
8. 门 15
9.  人 13
10. 地毯 29
11. 栏杆，栅栏  39，33
12. 底座  41
13. 柱子 43
14. 柜台（前台），展示柜  46，56,78,79,100
15. 冰箱 51
16. 楼梯  54，60,97
17. 坐垫，枕头等  40，58
18. 显示屏类  75,142，144
19. 动物  127
20. 自行车类别   128
21. 花瓶，瓶子  136
22. 垃圾桶  139

## **Yaoshun.Li. ade20k Modify 20 class**
1. 背景   0,2,3,6,9,10,14,17,21,22,26,27,28,35,37,38,42,44,47,48,50,52,53,61,62,66,68,69,77,80,81,82,83,84,85,86,87,88,90,91,92,93,94,95,96,99,103,104,105,106,107,108,109,110,112,123,114,115,117,116,120,121,123,126,129,132,133,134,135,137,138,140,141,143,145,146,148,149,150
2. 墙面,窗帘、绘画  1、19、23，59,64,101,102,124,131,147
3. 椅子，柜子类、桌子,底座   11，16，20，25，31，32，34，36，41，45，57,63,65,70,71,72,74，76，89，98,111,113,119,122,125,130
4. 植物,树,花瓶 18，5,67,73,136
5. 沙发 24、31
6. 地面 4, 12，30,55
7. 床 8, 118
8.  门 15
9.  人 13
10. 地毯 29
11. 栏杆，栅栏  39，33
12. 柱子 43
13. 柜台（前台），展示柜  46，56,78, 79,100
14. 冰箱 51
15. 楼梯  54，60,97
16. 坐垫，枕头等  40，58
17. 显示屏类  75,142，144
18. 动物  127
19. 自行车类别   128
20. 垃圾桶  139
    

## **Yaoshun.Li. ade20k 通过实验后的效果进行类别调整  15 class**
1. 背景   0,2,3,6,9,10,14,17,21,22,26,27,28,33,35,37,38,39,42,43,44,47,48,50,52,53,61,62,66,68,69,77,80,81,82,83,84,85,86,87,88,90,91,92,93,94,95,96,99,103,104,105,106,107,108,109,110,112,123,114,115,117,116,120,121,123,126,128,129,132,133,134,135,137,138,139,140,141,143,145,146,148,149,150
2. 墙面,窗帘、绘画  1,19,23,59,64,101,102,124,131,147
3. 椅子，柜子类、桌子,底座   11,16,20,25,31,32,34,36,41,45,46,56,57,63,65,70,71,72,74,76,78,79,89,98,100,111,113,119,122,125,130
4. 植物,树,花瓶 18,5,67,73,136
5. 沙发 24,31
6. 地面 4,12,30,55
7. 床   8,118
8.  门 15
9.  人 13
10. 地毯 29
11. 冰箱 51
12. 楼梯  54,60,97
13. 坐垫，枕头等  40,58
14. 显示屏类  75,142,144
15. 动物  127
    
---

## **合并步骤**
```
其中一些主要的合并功能使用python实现，主要的实现代码参考pick_script.py
```





## **ADE20K室内场景类别**
- art_gallery
- ball_pit
- bathroom  |  我们的数据中不需要这个类别
- bedroom
- booth_indoor
- closet
- conference_room
- corridor
- day_care_center
- elevator_interior
- gymnasium_indoor
- kiosk_indoor
- kitchen
- library_indoor
- living_room
- locker_room
- office
- bookshelf
- mews
- nook
- palace
- palace_hall
- establishment
- poolroom_home
- sandbox
- subway_interior
- platform
- television_studio
- indoor_procenium
- airplane_cabin
- airlock
- airport_ticket_counter
- alcove
- anechoic_chamber
- apse_indoor
- arcade
- archive
- arrival_gate_indoor
- art_gallery
- art_school
- artists_loft
- athletic_field_indoor
- atrium_home
- attic
- auto_mechanics_indoor
- badminton_court_indoor
- baggage_claim
- balcony_interior
- ballroom
- bank_indoor
- banquet_hall
- baptistry_indoor
- bar
- barbershop
- barrack
- basement
- basketball_court_indoor
- batting_cage_indoor
- bazaar_indoor
- beauty_salon
- beer_garden
- betting_shop
- biology_laboratory
- bistro_indoor
- bleachers_indoor
- bomb_shelter_indoor
- booth_indoor
- bow_window_indoor
- bowling_alley
- breakroom
- brewery_indoor
- bus_interior
- cabin_indoor
- cafeteria
- call_center
- canteen
- cardroom
- cargo_container_interior
- carport_indoor
- casino_indoor
- cathedral_indoor
- catwalk
- cavern_indoor
- chapel
- checkout_counter
- chicken_coop_indoor
- chicken_farm_indoor
- childs_room
- choir_loft_interior
- church_indoor
- circus_tent_indoor
- classroom
- clean_room
- booth
- clock_tower_indoor
- cloister_indoor
- coffee_shop
- computer_room
- conference_center
- conference_hall
- confessional
- control_room
- control_tower_indoor
- convenience_store_indoor
- corridor
- courtroom
- courtyard    | 犹豫中
- crosswalk
- library
- cybercafe
- dance_school
- delicatessen   | 犹豫中
- dentists_office
- departure_lounge
- diner_indoor
- dinette_home
- vehicle    | 犹豫中
- dining_car
- dining_room
- discotheque
- doorway_indoor
- dorm_room
- dressing_room
- driving_range_indoor
- drugstore
- door
- elevator_lobby
- entrance_hall
- escalator_indoor
- exhibition_hall
- factory_indoor
- fastfood_restaurant
- ferryboat_indoor
- firing_range_indoor
- fishmarket
- exterior
- fitting_room_interior
- flea_market_indoor
- florist_shop_indoor
- food_court
- foundry_indoor
- funeral_chapel
- funeral_home
- game_room
- garage_indoor
- gazebo_interior
- general_store_indoor
- geodesic_dome_indoor
- gift_shop
- great_hall
- greenhouse_indoor
- gymnasium_indoor
- hallway
- hangar_indoor
- hardware_store
- home_office
- home_theater
- hospital_room
- hot_tub_indoor
- hotel_breakfast_area
- hotel_room
- hunting_lodge_indoor
- ice_cream_parlor
- ice_skating_rink_indoor
- incinerator_indoor
- inn_indoor
- jacuzzi_indoor
- jail_indoor
- jail_cell
- jury_box
- kennel_indoor
- kindergarden_classroom
- kiosk_indoor
- kitchenette
- lab_classroom
- labyrinth_indoor
- landing
- laundromat   | 犹豫
- lecture_room
- legislative_chamber
- lido_deck_indoor
- limousine_interior
- liquor_store_indoor
- living_room
- lobby
- locker_room
- loft
- lookout_station_indoor
- lumberyard_indoor
- manufactured_home
- market_indoor
- martial_arts_gym
- medina
- mess_hall
- military_hospital
- mini_golf_course_indoor
- monastery_indoor
- mobile_home
- mosque_indoor
- movie_theater_indoor
- museum_indoor
- misc    |  杂项
- newsroom
- newsstand_indoor
- nuclear_power_plant_indoor
- nursery
- nursing_home
- observatory_indoor
- oil_refinery_indoor
- optician
- organ_loft_interior
- outhouse_indoor
- oyster_bar
- amphitheater_indoor
- questionable
- assembly_hall
- awning_deck
- back_porch
- backstairs_indoor
- bath_indoor
- bookshelf
- bunk_bed
- cocktail_lounge
- deck-house_boat_deck_house
- dining_area
- entranceway_indoor
- field_tent_indoor
- flatlet
- flume_indoor
- front_porch
- patio_indoor
- salon
- sanatorium
- scullery
- snack_bar
- store
- student_center
- study_hall
- sunroom
- ticket_window_indoor
- vestibule
- palace_hall
- pantry
- parking_garage_indoor
- parlor
- party_tent_indoor
- pedestrian_overpass_indoor
- planetarium_indoor
- playroom
- plaza
- podium_indoor
- poolroom_home
- porch
- portrait_studio
- ADE_train_00015610
- quonset_hut_indoor
- reading_room
- reception
- refectory
- restaurant
- restaurant_kitchen
- restroom_indoor
- roller_skating_rink_indoor
- schoolyard
- server_room
- sewing_room
- shopping_mall_indoor
- shower_room
- massage_room
- stage_indoor
- staircase
- subway_interior
- supermarket  |  杂项
- sushi_bar
- swimming_pool_indoor
- synagogue_indoor
- tearoom
- television_room
- television_studio
- tennis_court_indoor
- indoor_procenium
- ticket_window_indoor
- tobacco_shop_indoor
- trading_floor
- train_interior
- station
- utility_room
- vestry
- veranda
- veterinarians_office
- volleyball_court_indoor
- waiting_room
- warehouse_indoor
- washhouse_indoor
- wet_bar
- widows_walk_interior
- window_seat
- witness_stand
- workroom
- wrestling_ring_indoor
- youth_hostel
- art_gallery
- art_studio
- attic
- shop
- balcony_interior
- ballroom
- bank_indoor
- bar
- bistro_indoor
- bow_window_indoor
- room
- control_tower_indoor
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

def load_data():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
d = load_data()
reg = d["model"]
# f = pd.read_csv("cars_engage_2022.csv")
# reg = d["reg"]
# le_make = d["le_make"]
# le_model = d["le_model"]
# le_variant = d["le_variant"]
# le_price = d["le_price"]
# le_cylinders = d["le_cylinders"]
# le_drivetrain = d["le_drivetrain"]
# le_engine_Location = d["le_engine_Location"]
# le_fuel_Type = d["le_fuel_Type"]
# le_height = d["le_height"]
# le_length = d["le_length"]
# le_width = d["le_width"]
# le_body_Type = d["le_body_Type"]
# le_doors = d["le_doors"]
# le_mileage = d["le_mileage"]
# le_type = d["le_type"]
# le_wheelbase = d["le_wheelbase"]
# le_wheels_Size = d["le_wheels_Size"]
#
# make = ['Tata', 'Datsun', 'Renault', 'Maruti Suzuki', 'Hyundai', 'Premier',
#        'Toyota', 'Nissan', 'Volkswagen', 'Ford', 'Mahindra', 'Fiat',
#        'Honda', 'Jeep', 'Isuzu', 'Skoda', 'Audi', 'Dc', 'Mini',
#        'Volvo', 'Jaguar', 'Bmw', 'Land Rover', 'Porsche', 'Lexus',
#        'Maserati', 'Lamborghini', 'Bentley', 'Ferrari', 'Aston Martin',
#        'Bugatti', 'Bajaj', 'Icml', 'Force', 'Mg', 'Kia',
#        'Land Rover Rover', 'Mitsubishi', 'Maruti Suzuki R']
# model=['Nano Genx', 'Redi-Go', 'Kwid', 'Eeco', 'Alto K10', 'Go',
#        'Celerio Tour', 'Santro', 'Tiago', 'Celerio X', 'Ignis', 'Triber',
#        'Rio', 'Etios Liva', 'Micra Active', 'Bolt', 'Xcent Prime',
#        'Dzire Tour', 'Elite I20', 'Aura', 'Polo', 'Dzire', 'Freestyle',
#        'Ameo', 'Aspire', 'Platinum Etios', 'Etios Cross', 'Verito Vibe',
#        'Urban Cross', 'Glanza', 'Avventura', 'Jazz', 'Compass Trailhawk',
#        'Mu-X', 'Alturas G4', 'Tiguan', 'Cr-V', 'Superb Sportline', 'A3',
#        'Mercedes-Benz B-Class', 'Mercedes-Benz Cla-Class', 'Kodiaq',
#        'Avanti', 'Q3', 'Cooper 5 Door', 'Convertible', 'Xc40', 'Clubman',
#        'A4', 'John Cooper Works', 'Xe', 'Xf', 'A3 Cabriolet', 'A6', 'X3',
#        'Discovery Sport', 'S90', 'S5', 'X5', 'Mustang', 'Grand Cherokee',
#        'Mercedes-Benz E-Class Cabriolet', 'M2 Competition', '718',
#        'Mercedes-Benz Gls', 'Land Cruiser Prado', 'Rx 450H', 'Rs5',
#        '7-Series', 'Q8', 'Mercedes-Benz S-Class', 'Levante',
#        'Mercedes-Benz G-Class', 'A8 L', 'Granturismo', 'Quattroporte',
#        'Lc 500H', 'Mercedes-Benz Maybach', 'Panamera', 'Lx 450D',
#        'Mercedes-Benz S-Class Cabriolet', 'R8', 'Urus', 'Continental Gt',
#        'Portofino', 'Bentayga', 'Db 11', '458 Speciale',
#        'Rolls-Royce Ghost Series Ii', 'Rolls-Royce Wraith', 'Mulsanne',
#        'Rolls-Royce Cullinan', 'Rolls-Royce Phantom Coupe', 'Chiron',
#        'Qute (Re60)', 'Alto', 'S-Presso', 'Celerio', 'Grand I10 Prime',
#        'Kuv100 Nxt', 'Swift', 'Altroz', 'Extreme', 'Tigor', 'Zest',
#        'Amaze', 'Gypsy', 'Venue', 'Nexon', 'Linea', 'Bolero Power Plus',
#        'Vitara Brezza', 'I20 Active', 'Ecosport', 'Duster', 'Verna',
#        'Xuv300', 'Lodgy', 'Vento', 'E2O Plus', 'Tigor Ev', 'Brv', 'Thar',
#        'Gurkha', 'Xl6', 'Abarth Avventura', 'Tuv300 Plus', 'Marazzo',
#        'Scorpio', 'Monte Carlo', 'Xuv500', 'E Verito', 'Hexa',
#        'Innova Crysta', 'Compass', 'Corolla Altis', 'Civic', 'Zs Ev',
#        'Carnival', 'Superb', 'V40', 'Fortuner', 'Endeavour',
#        'Cooper 3 Door', 'Kodiaq Scout', 'X1', 'S60', '3-Series',
#        'S60 Cross Country', 'Q5', 'Range Evoque', 'Mercedes-Benz E-Class',
#        'Xc60', 'X4', 'Wrangler', 'Mercedes-Benz C-Class Cabriolet',
#        'Z4 Roadster', 'Mercedes-Benz V-Class', 'Q7',
#        'Range Evoque Convertible', 'Range Velar',
#        'Mercedes-Benz E-Class All Terrain', 'Xc90', 'Range Sport', 'Xj',
#        'Cayenne', 'Ghibli', 'Land Cruiser', 'Rs7', 'Range', 'Grancabrio',
#        'Mercedes-Benz Amg-Gt', 'Mercedes-Benz Amg Gt 4-Door Coupe',
#        'Huracan', '488 Gtb', 'Gtc4 Lusso', 'Aventador',
#        'Rolls-Royce Dawn', 'Rolls-Royce Drophead Coupe', 'Omni', 'Go+',
#        'Punto Evo Pure', 'Figo', 'Baleno', 'Grand I10', 'Linea Classic',
#        'Sunny', 'Ertiga', 'Baleno Rs', 'Wr-V', 'Tuv300', 'S-Cross',
#        'Captur', 'Xylo', 'Seltos', 'Terrano', 'Safari Storme', 'Hector',
#        'Nexon Ev', 'Elantra', 'Tucson', 'Passat', 'Mercedes-Benz A-Class',
#        'V40 Cross Country', 'Countryman', 'Mercedes-Benz C-Class',
#        'Prius', 'Es', 'Nx 300H', 'F-Pace', 'V90 Cross Country',
#        'A5 Cabriolet', 'Mercedes-Benz Gle', 'Mercedes-Benz Cls', 'X7',
#        'M4', '911', 'Gtr', 'Vantage', 'Rapide', '812 Superfast',
#        'Alto 800 Tour', 'Grand I10 Nios', 'Xcent', 'Micra', 'Bolero',
#        'Ciaz', 'Rapid', 'Abarth Punto', 'Creta', 'Harrier',
#        'Dmax V-Cross', 'Outlander', 'Mercedes-Benz Gla-Class',
#        'Accord Hybrid', '5-Series', '6-Series', 'Macan', 'F-Type', 'M5',
#        'Lx 570', '458 Spider', 'Wagon', 'Tiago Nrg', 'Nuvosport', 'Kicks',
#        'Winger', 'Kona Electric', 'Camry', 'A5', 'Discovery', 'Ls 500H',
#        'Rolls-Royce Phantom', 'Punto Evo', 'Yaris', 'Octavia',
#        'Mercedes-Benz Glc', 'Cayenne Coupe', 'Verito', 'Pajero Sport',
#        'Flying Spur', 'City', 'Montero']
# var_iant= ['Xt', 'Xe', 'Emax Xm', 'Zx Mt Diesel', 'Zx Cvt Petrol',
#        '3.2 At']
# drive_train=['RWD (Rear Wheel Drive)', 'FWD (Front Wheel Drive)',
#        'AWD (All Wheel Drive)', '4WD']
# engin_location=['Rear, Transverse', 'Front, Transverse', 'Front, Longitudinal', 'Rear Mid, Transverse', 'Mid, Longitudinal',
#        'Mid, Transverse', 'Rear, Longitudinal']
# fu_el = ['Petrol', 'CNG', 'Diesel', 'CNG + Petrol', 'Hybrid', 'Electric']
# typ = ['Hatchback', 'MPV', 'MUV', 'SUV', 'Sedan', 'Crossover',
#        'Coupe', 'Convertible', 'Sports, Hatchback', 'Sedan, Coupe',
#        'Sports', 'Crossover, SUV', 'SUV, Crossover', 'Sedan, Crossover',
#        'Sports, Convertible', 'Pick-up', 'Coupe, Convertible']
# Type=['Manual', 'Automatic', 'AMT', 'CVT', 'DCT']
# wheel_size=['4 B X 12', '80/155R13', '155/80R13', '165/70 R14', '155 R13 LT',
#        '155/65R13', '155/70R13', '165/70R14',  '155/80 R13',
#        '175/65 R14', '175/65R 15', '185/65 R14', '185/65 R15',
#        '205/70R15', 'R15', '175/65R14', '185/60R15', '13x4.5B Steel',
#        '175/65 R15', '175/ 65 R14', '165/65R14', '165/80R14', '185/70R14',
#        '195/55R16', '175/60 R15', '175/70R14', '185/60R16', '185/65R15',
#        'R16', '175/70 R14', '185/60 R15', '195 / 55 R16', '195/55R15',
#        '175/70R15', '205/55R16', '175/65R15', '225 / 60 R17',
#        '265 / 60 R17', '245/70R17', '255/60 R18', '215/65R17',
#        '235/55R18', '235 / 65 R18', '215/55 R17', '205/55R17',
#        '225/45R17', 'R20', '215/65R16', '235/55R17', '235/55 R17',
#        '235/55 R19', '235/55 R18',
#        '43.66 cm/17" Light Alloy Wheels JCW Track Spoke, Black',
#        '205 / 55 R17', '225/55 R18', '225/60R19', '235/60 R18',
#        '245/45R18', '245/40R18', '255/50R19', '265 / 60 R18',
#        '265/50 R20', '295/45R20', '255/35 R18', 'R19', '235/40 R18', '19',
#        '21', '265/60R18', '235/60VR18', '265/35R19', '(245/50 R18',
#        '275/40 R19', '275/35 R20', '275/50 R20', '245/45 R19',
#        '285/40ZR19', '255/60 ZR18', '285/45R21', '265/60 R 18',
#        '255/45 R19', 'ZR20', '245/40 R21', '245/40R20', '245/45R19',
#        'ZR19', '235/35 R19', '245/35 R19', '275/40R20', 'R21', 'ZR 20',
#        '265/45R20', '285/30ZR20', 'R12', '145/80R12', '145/80 R13', '14',
#        '185/65R14', '165/80 R14', '185/80R14', '15', '195 / 65 R15',
#        '215 / 60 R16', '195 / 65 R15 Steel', '195/60 R16', '195/60R15',
#        '215/75R15', '16', '215 / 75 R 15', '205/60R16', '215/60R16',
#        '195/65R15', '205/50R17', '205/65 R16', '185/65 R16', 'R14',
#        '235/70R16', 'P235/70 R16', '245 / 70 R16', '205/55 R16',
#        '215/70R15', '215 65 R16', '215/60 R17', 'P215/75R15', '245/75R16',
#        'P235/65R17', '235/65R17', '17', 'P235/65R15', 'P235/60 R18',
#        'P235/65 R17', '235/70 R16', '205/65R16', '215/55R17',
#        '265 / 60 R16', '215/55 R16', '215/50 R17', '115/90R17', 'R17',
#        '265/65R17', 'R18', '265/65R18', '225/55R17', '225/50 R18',
#        '225/40R18', '235/40R18', '225/50 R17', '215/50R17', '235/60R18',
#        '245/45 R17', '245/35 R17', '18', '245/50 R19', '245/75R15',
#        '255 / 335 R19', '255/60R18', '255/55R19', '255/60R19',
#        '155/85R18', '235/65R20', '21 Inch', '275/45 R21', '20',
#        '255/55 ZR 19', '285 / 40 ZR21', '285/60R18', '275/35R20', '22',
#        '355/25 ZR21', '195/55 R15', '205/60/R16', '205/65R15',
#        '225/55 R17', '115/90R16', '205 / 55 R18', '225/50R17',
#        '245/35R19', '235/45 R18', '225/60R18', '245/45R20', '245/40 R18',
#        '265/45 R20', '255/50 R20', '225 / 45 R18', 'ZR21', '255/35 R20',
#        '315/35 ZR20', '175/60R15', '15x5.5j Alloy', '14x5.5j Steel',
#        '215/60R17', '235/65 R17', '235/50R18', '235/45R19', '225/60R17',
#        '255/55R18', '19 inches', '20 inches', '155/65R14', '145/80R13',
#        '195 R15 LT, 8PR Radial', '235/45R18', '255/55 R20', '255/55 R19',
#        '245/45RF20', '235/55R19', 'ZR 21', '275/40R19', '185/55R16']
# wheel_base=['2230 mm', '2348 mm', '2422 mm', '2442 mm', '2350 mm', '2360 mm',
#        '2450 mm', '2425 mm', '2400 mm', '2435 mm', '2636 mm', '2420 mm',
#        '2460 mm', '2470 mm', '2430 mm', '2525 mm', '2570 mm', '2469 mm',
#        '2490 mm', '2550 mm', '2630 mm', '2510 mm', '2520 mm', '2530 mm',
#        '2845 mm', '2865 mm', '2677 mm', '2660 mm', '2841 mm', '2637 mm',
#        '2699 mm', '2700 mm', '2603 mm',  '2702 mm', '2670 mm',
#        '2820 mm', '2495 mm', '2835 mm', '2960 mm', '3157 mm', '2595 mm',
#        '2912 mm', '2810 mm', '2741 mm', '2941 mm', '2811 mm', '2575 mm',
#        '2720 mm', '2915 mm', '2760 mm', '2693 mm', '2475 mm', '3075 mm',
#        '2790 mm', '3210 mm', '3110 mm', '2994 mm', '3035 mm', '3165 mm',
#        '2945 mm', '3004 mm', '2850 mm', '2890 mm', '3128 mm', '2942 mm',
#        '3171 mm', '2870 mm', '3365 mm', '2950 mm', '3100 mm', '2650 mm',
#        '3003 mm', '2746 mm', '2669 mm', '2992 mm', '2995 mm', '2600 mm',
#        '3295 mm', '3112 mm', '3266 mm', '3320 mm', '2710 mm', '1925 mm',
#        '2380 mm', '2385 mm', '2501 mm', '2541 mm', '2375 mm', '2500 mm',
#        '2498 mm', '2680 mm', '2794 mm', '2519 mm', '2579 mm', '2673 mm',
#        '2553 mm', '2258 mm', '2662 mm', '2750 mm', '2740 mm', '3040 mm',
#        '2552 mm', '2585 mm', '3060 mm', '2647 mm', '2745 mm', '2791 mm',
#        '2776 mm', '2774 mm', '2807 mm', '2681 mm', '3079 mm', '2874 mm',
#        '2864 mm', '3008 mm', '2840 mm', '3430 mm', '3200 mm', '2984 mm',
#        '2923 mm', '2895 mm', '2998 mm', '2922 mm', '3120 mm', '2951 mm',
#        '2620 mm', '2990 mm', '3570 mm', '1840 mm', '2555 mm', '2610 mm',
#        '2786 mm', '2646 mm', '3105 mm', '2812 mm', '2780 mm', '2704 mm',
#        '2989 mm', '2590 mm', '3095 mm', '2968 mm', '2855 mm', '3070 mm',
#        '2622 mm', '2964 mm', '3488 mm', '2825 mm', '3125 mm', '3820 mm',
#        '2688 mm', '2873 mm', '2800 mm', '3066 mm']
# city_mileage=['?23.6 km/litre', '21.38 km/litre', '25.17 km/litre',
#        '12 km/litre', '11 km/litre', '14 km/litre', '19 km/litre',
#        '20.6 km/litre', '23 km/litre', '23.84 km/litre', '20.89 km/litre',
#        '15.1 km/litre', '13 km/litre', '20 km/litre', '20.3 km/litre',
#        '19.49 km/litre', '22,95 km/litre', '17,57 km/litre',
#        '14.6 km/litre', '16.3 km/litre', '13.3 km/litre', '18.4 km/litre',
#        '17 km/litre', '28,4 km/litre', '28.4 km/litre', '15.3 km/litre',
#        '13,6 km/litre', '20.32 km/litre', '13.6 km/litre', '15 km/litre',
#        '16,78 km/litre', '18.1 km/litre', '16.78 km/litre', '18 km/litre',
#        '11.2 km/litre', '11.3 km/litre', '13.8 km/litre', '12.4 km/litre',
#        '9.5 km/litre', '11.5 km/litre', '11.04 km/litre', '13.9 km/litre',
#        '8 km/litre', '10.3 km/litre', '12.3 km/litre', '5.7 km/litre',
#        '19.2 km/litre', '16 km/litre', '11,44 km/litre', '10 km/litre',
#        '12.8 km/litre', '4.5 km/litre', '12.5-12.7 km/litre',
#        '9 km/litre', '6.5 km/litre', '7 km/litre', '18.8 km/litre',
#        '13.5 km/litre', '5.4 km/litre', '5.2 km/litre', '4.6 km/litre',
#        '7.81 km/litre', '7.1 km/litre', '7.8 km/litre', '5.3 km/litre',
#        '6 km/litre', '9,6 km/litre', '10.5 km/litre', '4.7 km/litre',
#        '6.2 km/litre', '10.2 km/litre', '24 km/litre', '18.9 km/litre',
#        '15.5 km/litre', '22.25 km/litre', '12.6 km/litre', '8.6 km/litre',
#        '24.12 km/litre', '13.2 km/litre', '11,3 km/litre',
#        '17.2 km/litre', '9.4 km/litre', '21.19 km/litre', '19.9 km/litre',
#        '21.04 km/litre', '12.1 km/litre', '80 km/litre', '16.2 km/litre',
#        '18.49 km/litre', '8.1 km/litre', '110 km/litre', '18.2 km/litre',
#        '10.6 km/litre', '10.1 km/litre', '16,8 km/litre',
#        '12.55 km/litre', '12,55 km/litre', '7.7 km/litre', '9.3 km/litre',
#        '15.71 km/litre', '15.6 km/litre', '8.5 km/litre', '7.45 km/litre',
#        '11.7 km/litre', '17.5 km/litre', '15.68 km/litre',
#        '12.63 km/litre', '5 km/litre', '5.9 km/litre', '5.6 km/litre',
#        '4.4 km/litre', '8.69 km/litre', '4 km/litre', '5.8 km/litre',
#        '3.2 km/litre', '3 km/litre', '3.6 km/litre', '7.9 km/litre',
#        '4.38 km/litre', '20.62 km/litre', '27.39 km/litre',
#        '21.4 km/litre', '11.4 km/litre', '25.5 km/litre',
#        '23.65 km/litre', '10.7 km/litre', '17.1 km/litre',
#        '10.8 km/litre', '13,93 km/litre', '13.1 km/litre',
#        '16,38 km/litre', '13,3 km/litre', '16.38 km/litre',
#        '12.03 km/litre', '13.05 km/litre', '17.8 km/litre',
#        '21.27 km/litre', '7.32 km/litre', '11.6 km/litre', '9.1 km/litre',
#        '20.7 km/litre', '15.7 km/litre', '19.5 km/litre',
#        '28.09 km/litre', '21.56 km/litre', '26.82 km/litre',
#        '26032 km/litre', '26.32 km/litre', '14.5 km/litre',
#        '15.29 km/litre', '17.01 km/litre', '7.3 km/litre',
#        '13.4 km/litre', '21 km/litre', '8.4 km/litre', '15.01 km/litre',
#        '4.45 km/litre', '12.05 km/litre', '15.8 km/litre',
#        '21,2 km/litre', '2 km/litre', '14.3 km/litre', '22.6 km/litre',
#        '8.25 km/litre']
# wid_th=['1750 mm', '1560 mm', '1579 mm', '1580 mm', '1475 mm', '1490 mm',
#        '1636 mm', '1600 mm', '1645 mm', '1677 mm', '1647 mm', '1690 mm',
#        '1739 mm', '1570 mm', '1695 mm', '1665 mm', '1660 mm', '1710 mm',
#        '1734 mm', '1680 mm', '1682 mm', '1735 mm', '1737 mm', '1704 mm',
#        '1705 mm', '1740 mm', '1706 mm', '1745 mm', '1694 mm', '1818 mm',
#        '1860 mm', '1960 mm', '1839 mm', '1855 mm', '1864 mm', '1796 mm',
#        '1786 mm', '1777 mm', '2120 mm', '1831 mm', '1727 mm', '1863 mm',
#        '1801 mm', '1842 mm', '1850 mm', '2091 mm', '1899 mm', '1793 mm',
#        '1874 mm', '1881 mm', '2069 mm', '1879 mm', '1843 mm', '2218 mm',
#        '2080 mm', '1943 mm', '1954 mm', '1854 mm', '1934 mm', '1982 mm',
#        '1885 mm', '1895 mm',  '2169 mm', '2142 mm', '1968 mm',
#        '2158 mm', '1931 mm', '1945 mm', '1915 mm', '1948 mm', '1920 mm',
#        '1937 mm', '1980 mm', '2029 mm', '2181 mm', '2226 mm', '1910 mm',
#        '1998 mm', '1865 mm', '1951 mm', '1947 mm', '2208 mm', '2000 mm',
#        '1987 mm', '2038 mm', '1312 mm', '1520 mm', '1755 mm', '1540 mm',
#        '1770 mm', '1811 mm', '1730 mm', '1790 mm', '1760 mm', '1765 mm',
#        '1822 mm', '1729 mm', '1821 mm', '1751 mm', '1699 mm', '1575 mm',
#        '1726 mm', '1820 mm', '1775 mm', '1835 mm', '1866 mm', '1890 mm',
#        '1903 mm', '1830 mm', '1799 mm', '1809 mm', '1985 mm', '2041 mm',
#        '1869 mm', '1882 mm', '2060 mm', '2058 mm', '2097 mm', '1898 mm',
#        '1996 mm', '1902 mm', '1918 mm', '1877 mm', '1810 mm', '2024 mm',
#        '1459 mm', '1928 mm', '1900 mm', '2145 mm', '1852 mm', '2140 mm',
#        '2220 mm', '1983 mm', '2100 mm', '1911 mm', '2034 mm', '2073 mm',
#        '2056 mm', '1939 mm', '2007 mm', '1953 mm', '1924 mm', '1933 mm',
#        '1952 mm', '2030 mm', '1990 mm', '1410 mm', '1635 mm', '1687 mm',
#        '1785 mm', '1813 mm', '1800 mm', '1965 mm', '1832 mm', '1780 mm',
#        '2022 mm', '1783 mm', '1857 mm', '1.845 mm', '1845 mm', '2175 mm',
#        '2052 mm', '2157 mm', '1870 mm', '2153 mm', '1929 mm', '1971 mm',
#        '1894 mm', '1804 mm', '1849 mm', '1923 mm', '2042 mm', '1620 mm',
#        '1675 mm', '1905 mm', '1840 mm', '2200 mm', '1814 mm', '2194 mm',
#        '1815 mm', '2207 mm', '1875 mm']
# len_gth=['3164 mm', '3429 mm', '3731 mm', '3675 mm', '3545 mm', '3620 mm',
#        '3788 mm', '3600 mm', '3695 mm', '3610 mm', '3765 mm', '3746 mm',
#        '3700 mm', '3990 mm', '3970 mm', '3884 mm', '3801 mm', '3825 mm',
#        '3995 mm', '3985 mm', '3971 mm', '3954 mm', '4369 mm', '3895 mm',
#        '3991 mm', '3989 mm', '3955 mm', '4398 mm', '4825 mm', '4850 mm',
#        '4486 mm', '4592 mm', '4861 mm', '4456 mm', '4393 mm', '4630 mm',
#        '4565 mm', '4388 mm', '3982 mm', '3850 mm', '4425 mm', '4253 mm',
#        '4726 mm', '3874 mm', '4691 mm', '4961 mm', '5252 mm', '5067 mm',
#        '4421 mm', '4933 mm', '4657 mm', '4600 mm', '4963 mm', '4718 mm',
#        '4922 mm', '4784 mm', '4828 mm', '4846 mm', '4703 mm', '4461 mm',
#        '4379 mm', '5120 mm', '5146 mm', '4840 mm', '5000 mm', '4649 mm',
#        '5219 mm', '5052 mm', '5116 mm', '5246 mm', '5027 mm', '5003 mm',
#        '4763 mm', '4817 mm', '5302 mm', '4881 mm', '5262 mm', '4770 mm',
#        '5453 mm', '5049 mm', '5199 mm', '5080 mm', '4440 mm', '5112 mm',
#        '4807 mm', '4569 mm', '5141 mm', '5140 mm', '4385 mm', '4571 mm',
#        '5399 mm', '5569 mm', '5269 mm', '5575 mm', '5341 mm', '5612 mm',
#        '4544 mm', '2752 mm', '3445 mm', '3565 mm', '3840 mm', '3992 mm',
#        '4010 mm', '3994 mm', '4596 mm', '4494 mm', '3998 mm', '4315 mm',
#        '4498 mm', '4390 mm', '3390 mm', '4453 mm', '3920 mm', '4342 mm',
#        '4445 mm', '3983 mm', '4400 mm', '4585 mm', '5118 mm', '4413 mm',
#        '4247 mm', '4788 mm', '4735 mm', '4395 mm', '4620 mm', '4656 mm',
#        '4314 mm', '5115 mm', '4795 mm', '4903 mm', '3821 mm', '4697 mm',
#        '4439 mm', '4477 mm', '4635 mm', '4633 mm', '4637 mm', '4663 mm',
#        '4371 mm', '5063 mm', '4879 mm', '4688 mm', '4752 mm', '4882 mm',
#        '4686 mm', '4324 mm', '5370 mm', '4370 mm', '4797 mm', '4950 mm',
#        '5255 mm', '4918 mm', '4926 mm', '4971 mm', '5012 mm', '4999 mm',
#        '5200 mm', '4546 mm', '4551 mm', '5054 mm', '4459 mm', '4520 mm',
#        '4568 mm', '4780 mm', '5285 mm', '5842 mm', '3370 mm', '3987 mm',
#        '3941 mm', '4560 mm', '4455 mm', '4265 mm', '3999 mm', '4300 mm',
#        '4329 mm', '4331 mm', '4655 mm', '4475 mm', '4767 mm', '4292 mm',
#        '4299 mm', '4756 mm', '4540 mm', '4975 mm', '4.64 mm', '4640 mm',
#        '4731 mm', '4733 mm', '4924 mm', '4988 mm', '5151 mm', '4671 mm',
#        '4519 mm', '4710 mm', '4465 mm', '5019 mm', '3430 mm', '3395 mm',
#        '3805 mm', '4107 mm', '4221 mm', '4490 mm', '4270 mm', '4598 mm',
#        '5295 mm', '4695 mm', '4424 mm', '4907 mm', '4894 mm', '5091 mm',
#        '4696 mm', '4470 mm', '4482 mm', '4956 mm', '3655 mm', '3793 mm',
#        '4384 mm', '5458 mm', '4180 mm', '4885 mm', '5235 mm', '6092 mm',
#        '4670 mm', '4658 mm', '4931 mm', '4277 mm', '5299 mm', '4900 mm']
# hei_ght = ['1652 mm', '1541 mm', '1490 mm', '1800 mm', '1475 mm', '1460 mm',
#        '1507 mm', '1560 mm', '1535 mm', '1595 mm', '1643 mm', '1730 mm',
#        '1510 mm', '1530 mm', '1562 mm', '1520 mm', '1555 mm', '1505 mm',
#        '1469 mm', '1515 mm', '1570 mm', '1483 mm', '1525 mm', '1540 mm',
#        '1542 mm', '1544 mm', '1657 mm', '1840 mm', '1845 mm', '1.845 mm',
#        '1672 mm', '1679 mm', '1689 mm', '1416 mm', '1557 mm', '1432 mm',
#        '1200 mm', '1608 mm', '1425 mm', '1415 mm', '1441 mm', '1427 mm',
#        '1414 mm', '1457 mm', '1409 mm', '1455 mm', '1678 mm', '1727 mm',
#        '1443 mm', '1384 mm', '1745 mm', '1391 mm', '1802 mm', '1749 mm',
#        '1398 mm', '1410 mm', '1281 mm', '1850 mm', '1880 mm', '1700 mm',
#         '1479 mm', '1481 mm', '1740 mm', '1494 mm', '1411 mm',
#        '1938 mm', '1969 mm', '1485 mm', '1353 mm', '1345 mm', '1498 mm',
#        '1865 mm', '1417 mm', '1252 mm', '1638 mm', '1401 mm', '1320 mm',
#        '1742 mm', '1250 mm', '1203 mm', '1550 mm', '1521 mm', '1835 mm',
#        '1598 mm', '1212 mm', '1549 mm', '1655 mm', '1523 mm', '1885 mm',
#        '1537 mm', '1501 mm', '1875 mm', '1590 mm', '1607 mm', '1487 mm',
#        '1977 mm', '1640 mm', '1647 mm', '1695 mm', '1445 mm', '1617 mm',
#        '1697 mm', '1467 mm', '1585 mm', '1666 mm', '1930 mm', '2055 mm',
#        '2075 mm', '1812 mm', '1774 mm', '1874 mm', '1995 mm', '1466 mm',
#        '1785 mm', '1791 mm', '1795 mm', '1433 mm', '1644 mm', '1755 mm',
#        '1420 mm', '1837 mm', '1665 mm', '1612 mm', '1545 mm', '1484 mm',
#        '1429 mm', '1539 mm', '1659 mm', '1649 mm', '1474 mm', '1658 mm',
#        '1621 mm', '1838 mm', '1304 mm', '1901 mm', '1609 mm', '1776 mm',
#        '1803 mm', '1696 mm', '1673 mm', '1705 mm', '1461 mm', '1910 mm',
#        '1419 mm', '1877 mm', '1827 mm', '1868 mm', '1288 mm', '1284 mm',
#        '1259 mm', '1447 mm', '1165 mm', '1213 mm', '1383 mm', '1136 mm',
#        '1502 mm', '1500 mm', '1690 mm', '1685 mm', '1601 mm', '1817 mm',
#        '1839 mm', '1619 mm', '1895 mm', '1645 mm', '1671 mm', '1922 mm',
#        '1760 mm', '1465 mm', '1660 mm', '1456 mm', '1458 mm', '1470 mm',
#        '1442 mm', '1426 mm', '1651 mm', '1543 mm', '1386 mm', '1772 mm',
#        '1435 mm', '1805 mm', '1300 mm', '1299 mm', '1370 mm', '1273 mm',
#        '1360 mm', '1276 mm', '1630 mm', '1706 mm', '1855 mm', '1710 mm',
#        '1464 mm', '1369 mm', '1538 mm', '1624 mm', '1308 mm', '1311 mm',
#        '1473 mm', '1211 mm', '1675 mm', '1620 mm', '1587 mm', '1870 mm',
#        '2670 mm', '1846 mm', '1450 mm', '1495 mm', '1476 mm', '1676 mm',
#        '1488 mm', '1900 mm']
# cyl_inders=[ 2.,  3.,  4.,  5.,  6.,  8., 12., 10., 16.]
def show_predict_page():
    st.title("Welcome to price prediction page")
    st.write("Fill the data to predict the price of car")
    # comname = st.selectbox("Enter the name of Maker", make)
    # comname = le_make[make.index(comname)]
    # modelname = st.selectbox("Enter the name of the car", f["Model"].unique())
    # modelname = le_model[model.index(modelname)]
    # varname = st.selectbox("Enter the variant", f["Variant"].unique())
    # varname = le_variant[var_iant.index(varname)]
    # price = st.selectbox("Enter the showroom price", f['Ex-Showroom_Price'].unique())
    # price = le_price[f["Ex-Showroom Price"].unique().index(price)]
    # cylinders = st.selectbox("Enter the number of cylinders", f['Cylinders'].unique())
    # drivetrain = st.selectbox("Enter the Drivetrain of the car", f['Drivetrain'].unique())
    # drivetrain = le_drivetrain[drive_train.index(drivetrain)]
    # engine_Location = st.selectbox("Enter the Engine location", f['Engine_Location'].unique())
    # engine_Location = le_engine_Location[engin_location.index(engine_Location)]
    # fuel_Type = st.selectbox("Enter the Fuel Type", f['Fuel_Type'].unique())
    # fuel_Type = le_fuel_Type[fu_el.index(fuel_Type)]
    # height = st.selectbox("Enter the Type of fuel",f['Height'].unique())
    # height = le_height[hei_ght.index(height)]
    # width = st.selectbox("Enter the  Width", f['Width'].unique())
    # width = le_width[wid_th.index(width)]
    # length = st.selectbox("Enter the  Height", f['Length'].unique())
    # length = le_length[len_gth.index(length)]
    # body_Type = st.selectbox("Enter the  Body Type", f['Body_Type'].unique())
    # body_Type = le_body_Type[typ.index(body_Type)]
    # doors = st.selectbox("Enter the  number of doors", f['Doors'].unique())
    # doors = le_doors[f["Doors"].index(doors)]
    # city_Mileage = st.selectbox("Enter the  Mileage", f['City_Mileage'].unique())
    # city_Mileage = le_mileage[city_mileage.index(city_Mileage)]
    # type = st.selectbox("Enter the  Type ", f['Type'].unique())
    # type = le_type[type.index(type)]
    # wheelbase = st.selectbox("Enter the  wheelbase", f['Wheelbase'].unique())
    # wheelbase = le_wheelbase[wheel_base.index(wheelbase)]
    # wheels_Size = st.selectbox("Enter the  Wheel size", f['Wheels_Size'].unique())
    # wheels_Size = le_wheels_Size[wheel_size.index(wheels_Size)]
    year = st.number_input("Enter the year when car was bought",min_value=2000, max_value=2022, value=2010)
    km = st.number_input("Enter the distance it travelled", min_value=0, value=50000)
    fuel = st.selectbox("Enter the type of fuel",["Petrol", "Diesel", "CNG"])
    if fuel == "Petrol":
        fuel = 0
    elif fuel == "Diesel":
        fuel = 1
    else:
        fuel = 2
    seller = st.selectbox("Enter type of seller", ["Dealer", "Individual"])
    if seller == "Dealer":
        seller = 0
    else:
        seller = 1
    transmission = st.selectbox("Enter transmission type",["Manual", "Automatic"])
    if transmission == "Manual":
        transmission = 0
    else:
        transmission = 1
    present = st.number_input("Enter the current price of the car", value=200000)
    ok = st.button("Predict")
    if ok:
        X = np.array([[year,km, fuel,seller, transmission, present]])
        X = X.astype(float)
        selling_price = reg.predict(X)
        st.write(f"The predicted price is {selling_price[0].round(2) - 100000}")
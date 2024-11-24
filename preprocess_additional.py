import pandas as pd
import math
import os


def dms_to_decimal(dms_str):
    degrees, minutes, seconds = 0, 0, 0
    parts = dms_str.replace("°", " ").replace("'", " ").split()
    if len(parts) >= 3:
        d, m, s = parts[:3]
        degrees = float(d)
        minutes = float(m)
        seconds = float(s)

    return degrees + minutes / 60 + seconds / 3600


def calculate_solar_position(row, time):
    # 將時間轉換為小時格式
    time_decimal = int(time[:2]) + int(time[2:]) / 60

    # 將時間欄位轉為字符串，確保有前導零
    sunrise_str = f"{int(row['日出時刻']):04d}"
    sunset_str = f"{int(row['日落時刻']):04d}"
    noon_str = f"{int(row['太陽過中天']):04d}"

    sunrise_decimal = int(sunrise_str[:2]) + int(sunrise_str[2:]) / 60
    sunset_decimal = int(sunset_str[:2]) + int(sunset_str[2:]) / 60
    noon_decimal = int(noon_str[:2]) + int(noon_str[2:]) / 60

    noon_altitude = row["中天仰角"]
    sunrise_azimuth = row["日出方位角"]
    sunset_azimuth = row["日落方位角"]
    device_id = row["裝置ID"]
    latitude_dms = row["緯度(N)"]

    # 將緯度轉換為十進制度
    latitude = dms_to_decimal(latitude_dms)
    phi = math.radians(latitude)  # 轉為弧度

    # 計算太陽赤緯角 (delta)
    # 根據中天高度角公式: h_noon = 90 - latitude + delta => delta = h_noon + latitude - 90
    delta = math.radians(noon_altitude + latitude - 90)

    # 計算時角 (H)
    if time_decimal < noon_decimal:
        H = -15 * (noon_decimal - time_decimal)
    else:
        H = 15 * (time_decimal - noon_decimal)
    H_rad = math.radians(H)

    # 計算仰角 (h)
    try:
        h_rad = math.asin(math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(H_rad))
        h = math.degrees(h_rad)
    except:
        h = 0.0  # 當太陽在地平線下時，仰角設為0

    # 插值計算方位角 (A)
    # 考慮太陽從日出到日落的移動，使用線性插值
    if sunset_decimal != sunrise_decimal:
        A = sunrise_azimuth + (sunset_azimuth - sunrise_azimuth) * (
            (time_decimal - sunrise_decimal) / (sunset_decimal - sunrise_decimal)
        )
    else:
        A = sunrise_azimuth  # 避免除以零

    return round(h, 2), round(A, 2), device_id


def align_time(total_minute, direction="up"):
    if direction == "up":
        return ((total_minute + 9) // 10) * 10
    elif direction == "down":
        return (total_minute // 10) * 10
    else:
        return total_minute


def main():
    project_root = os.getcwd()
    additional_data_path = os.path.join(project_root, "AdditionalData", "additional_data_17.csv")
    output_csv_dir = os.path.join(project_root, "AdditionalTrainData")
    output_csv_path = os.path.join(output_csv_dir, "AdditionalTrainData_17.csv")
    os.makedirs(output_csv_dir, exist_ok=True)

    df = pd.read_csv(additional_data_path, dtype={"日出時刻": str, "太陽過中天": str, "日落時刻": str, "裝置ID": str})

    results = []
    for _, row in df.iterrows():
        date = str(row["日期"])
        device_id = str(row["裝置ID"])
        latitude = dms_to_decimal(row["緯度(N)"])
        sunrise_time = int(row["日出時刻"])
        sunset_time = int(row["日落時刻"])

        sunrise_hour = sunrise_time // 100
        sunrise_minute = sunrise_time % 100
        sunset_hour = sunset_time // 100
        sunset_minute = sunset_time % 100
        sunrise_total = sunrise_hour * 60 + sunrise_minute
        sunset_total = sunset_hour * 60 + sunset_minute

        aligned_sunrise = align_time(sunrise_total, direction="up")
        aligned_sunset = align_time(sunset_total, direction="down")

        for total_minute in range(aligned_sunrise, aligned_sunset + 1, 10):
            hour = total_minute // 60
            minute = total_minute % 60
            time = f"{hour:02d}{minute:02d}"

            altitude, azimuth, device = calculate_solar_position(row, time)

            datetime_with_id = f"{date}{time}{device}"

            results.append({"Serial": datetime_with_id, "ElevationAngle": altitude, "Azimuth": azimuth})

    output_df = pd.DataFrame(results)
    output_df = output_df[["Serial", "ElevationAngle", "Azimuth"]]
    output_df.to_csv(output_csv_path, index=False, sep=",", header=True, encoding="utf-8-sig")
    print("Data processing complete.")


if __name__ == "__main__":
    main()

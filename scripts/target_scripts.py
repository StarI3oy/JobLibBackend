# -*- coding: utf-8 -*-
import datetime
import glob
import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb


def create_target_set(target_name, how_to_shift, hours=48):
    """
    Функция для создания таргет-датасета

    in->:
    target_name - название таргета
    how_to_shift - метод сдвига (по умолчанию "2")
    hours - количество часов для сдвига варианты - 48, 72, 96

    out<-:
    Target_KS - результирующий датафрейм в виде таргета
    """

    Target_KS = pd.read_csv(
        f"{target_name}P_in_out.csv", sep=";"
    )  # Датафрейм с таргетированным давлением
    """
    Обработка колонок данных таргетного датафрейма в нужном формате
    """
    Target_KS["DateTime"] = pd.to_datetime(Target_KS["DateTime"])
    Target_KS["dayofweek"] = Target_KS["DateTime"].dt.dayofweek
    Target_KS["P_in"] = Target_KS["P_in"].str.replace(",", ".").astype("float32")  #
    Target_KS["P_out"] = Target_KS["P_out"].str.replace(",", ".").astype("float32")
    Target_KS["E"] = Target_KS["E"].str.replace(",", ".").astype("float32")
    Target_KS["T_in"] = Target_KS["T_in"].str.replace(",", ".").astype("float32")
    Target_KS["T_out"] = Target_KS["T_out"].str.replace(",", ".").astype("float32")
    Target_KS["Tnv"] = Target_KS["Tnv"].str.replace(",", ".").astype("float32")

    """
    Преобразование колонки PNA из типа object в тип float32
    """
    if Target_KS["PNA"].dtype == "object":
        Target_KS["PNA"] = Target_KS["PNA"].str.replace(",", ".").astype("float32")
    else:
        Target_KS["PNA"] = Target_KS["PNA"].astype("float32")

    """
    Создание чего-то до сели неизведанного, которое будет использоваться в сдвиге 
    """
    abs = pd.DataFrame()
    abs["DateTime"] = pd.date_range(
        Target_KS["DateTime"].min(),
        periods=(
            (Target_KS["DateTime"].max() - Target_KS["DateTime"].min()).seconds / 3600
            + (Target_KS["DateTime"].max() - Target_KS["DateTime"].min()).days * 24
            + 1
        ),
        freq="h",
    )
    Target_KS = pd.merge(abs, Target_KS, how="left", on="DateTime")

    Target_KS = (
        Target_KS.ffill()
    )  # Тут заполнение пропусков чтобы дискретность была час

    """
    Способ сдвига данных (по умолчанию работает только "2")
    """
    if how_to_shift == "1":
        Target_KS.loc[Target_KS["dayofweek"] == 0, "DateTime"] -= pd.DateOffset(
            hours=72
        )
        Target_KS.loc[Target_KS["dayofweek"] == 1, "DateTime"] -= pd.DateOffset(
            hours=96
        )
        Target_KS.loc[Target_KS["dayofweek"] == 2, "DateTime"] -= pd.DateOffset(
            hours=48
        )
        Target_KS.loc[Target_KS["dayofweek"] == 3, "DateTime"] -= pd.DateOffset(
            hours=48
        )
        Target_KS.loc[Target_KS["dayofweek"] == 4, "DateTime"] -= pd.DateOffset(
            hours=48
        )
        Target_KS.loc[Target_KS["dayofweek"] == 5, "DateTime"] -= pd.DateOffset(
            hours=48
        )
        Target_KS.loc[Target_KS["dayofweek"] == 6, "DateTime"] -= pd.DateOffset(
            hours=48
        )
        Target_KS["dayofweek"] = Target_KS["DateTime"].dt.dayofweek
        Target_KS.columns = [
            f"{c}_" + target_name if c != "DateTime" else c for c in Target_KS
        ]
        Target_KS = Target_KS.drop(
            Target_KS[(Target_KS["DateTime"].dt.year < 2021)].index
        )
        return Target_KS
    elif how_to_shift == "2":
        Target_KS["DateTime"] -= pd.DateOffset(hours=hours)
        Target_KS["dayofweek"] = Target_KS["DateTime"].dt.dayofweek
        Target_KS.rename(
            columns={
                "P_in": "Pin_target_shift_" + str(hours) + "h",
                "P_out": "Pout_target_shift_" + str(hours) + "h",
            },
            inplace=True,
        )
        Target_KS.columns = [
            f"{c}_" + target_name if c != "DateTime" else c for c in Target_KS
        ]
        Target_KS = Target_KS.drop(
            Target_KS[(Target_KS["DateTime"].dt.year < 2021)].index
        )
        return Target_KS
    else:
        Target_KS.columns = [
            f"{c}_" + target_name if c != "DateTime" else c for c in Target_KS
        ]
        return Target_KS


def create_cross_target_agg(Target_List):
    """
    Лепим другие таргеты
    Ебашим лаги

    in->:
    Target_list - список кс, представленный в виде ["КС-15","КС-16","КС-17","КС-19"] в изначальной комплектации
    out<-:
    result_set - результирующий датасет
    """
    result_set = pd.DataFrame(columns=["DateTime"])
    for target_name in Target_List:
        Target_KS = pd.read_csv(target_name + "P_in_out.csv", sep=";")[
            ["DateTime", "P_in", "P_out"]
        ]
        Target_KS["DateTime"] = pd.to_datetime(Target_KS["DateTime"])
        Target_KS[target_name + "Pin"] = (
            Target_KS["P_in"].str.replace(",", ".").astype("float32")
        )
        Target_KS[target_name + "Pout"] = (
            Target_KS["P_out"].str.replace(",", ".").astype("float32")
        )
        Target_KS.drop(columns=["P_in", "P_out"], inplace=True)
        result_set = pd.merge(result_set, Target_KS, how="right", on="DateTime")
    col_list = result_set.columns
    for columns in col_list:
        if columns == "DateTime":
            continue
        result_set[columns + "_lag_48h"] = result_set[columns].shift(24)
        result_set[columns + "_lag_72h"] = result_set[columns].shift(36)
        result_set[columns + "_lag_96h"] = result_set[columns].shift(48)
        result_set[columns + "_lag_120h"] = result_set[columns].shift(60)
        result_set.drop(columns=columns, inplace=True)
    return result_set


def correct_data_into_value(date):
    """
    Функция для исправления формата дат в нужный формат данных (дата в строку)
    """
    if type(date) == type(datetime.datetime(2024, 2, 13, 0, 0)):
        return str(f"{date.day}.{date.month%12}")
    return date


def prepare_weather_dict_dataset():
    """
    Функция для формирования датасета погоды с привязкой к дате.
    Для этого достаем xlsx файлы из папки и образуем датафреймы

    out<-:
    weather_dfs - датафрейм с погодой по разным пунктам, соединенным LEFT JOIN по DateTime
    """
    weather_dict = {}
    for filename in os.listdir("weather_dfs/"):
        if filename.endswith(".xlsx"):
            weather_dict[filename[:-5]] = f"weather_dfs/{filename}"
    weather_dfs = {}
    for key, value in weather_dict.items():
        active_df = pd.ExcelFile(value)
        for i in active_df.sheet_names:
            if len(active_df.sheet_names) < 2:
                weather_dfs.update(
                    {
                        key: {
                            i: pd.read_excel(value, sheet_name=i).drop(
                                "Unnamed: 0", axis=1
                            )
                        }
                    }
                )
            else:
                for j in active_df.sheet_names:
                    weather_dfs.update(
                        {
                            f"{key}_{j}": {
                                j: pd.read_excel(value, sheet_name=j).drop(
                                    "Unnamed: 0", axis=1
                                )
                            }
                        }
                    )
    for i in weather_dfs.keys():
        for j in weather_dfs[i].keys():
            weather_dfs[i][j][weather_dfs[i][j].columns[1]] = pd.to_datetime(
                weather_dfs[i][j][weather_dfs[i][j].columns[1]]
            )
            weather_dfs[i][j] = (
                weather_dfs[i][j]
                .rename(columns={weather_dfs[i][j].columns[1]: "DateTime"})
                .select_dtypes(exclude=["object"])
            )
    return weather_dfs


def prepare_target_set(target_set, weather_set):
    """
    Предфинальная обработка данных (работа с датами и датасетом погоды)

    in->:
    target_set - датасет, образованный функцией prepare_dataset()
    weathere_set - датасет с погодой, образованный функцией prepare_weather_dict_dataset()
    out<-:
    target_set - предфинальный датасет, который можно использовать как
    исходныйЮ образованный из всех необходимых данных
    """
    target_set["Day"] = target_set["DateTime"].dt.day
    target_set["Month"] = target_set["DateTime"].dt.month
    target_set["Year"] = target_set["DateTime"].dt.year
    target_set["Hour"] = target_set["DateTime"].dt.hour
    target_set.columns = target_set.columns.str.replace(r"\n", "_", regex=True)
    #! DEBUG
    # for i in weather_set.keys():
    #     for j in weather_set[i].keys():
    #         buffer = weather_set[i][j].rename(
    #             lambda x: x + ("_" + i) * int(x != "DateTime"), axis="columns"
    #         )
    #         for columns in buffer.columns:
    #             if columns == "DateTime":
    #                 buffer["Day"] = buffer["DateTime"].dt.day
    #                 buffer["Month"] = buffer["DateTime"].dt.month
    #                 buffer["Year"] = buffer["DateTime"].dt.year
    #                 buffer["Hour"] = buffer["DateTime"].dt.hour
    #                 continue
    #             if buffer[columns].dtype == "float64":
    #                 buffer[columns] = buffer[columns].astype("float32")
    #         buffer.drop(columns=["DateTime"], inplace=True)
    #         target_set = pd.merge(
    #             target_set, buffer, on=["Year", "Month", "Day", "Hour"], how="left"
    #         )

    target_set.columns = target_set.columns.str.replace(r"\n", "_", regex=True)
    target_set[target_set.select_dtypes(include=["float64"]).columns] = target_set[
        target_set.select_dtypes(include=["float64"]).columns
    ].astype("float32")
    # for i in target_set.columns:
    #     if len(target_set[i].value_counts()) < 2:
    #         try:
    #             target_set.drop(columns=i, inplace=True)
    #         except KeyError:
    #             continue
    return target_set


def prepare_shift_and_lags(target_set):
    """
    Образуем ролинги и сдвигаем по 1, 3, 7 дням
    in->:
    target_set - исходный набор данных
    out<-:
    target_set - набор данных доработанный ролингами и лагами
    """
    col_list = target_set.columns
    for columns in col_list:
        if columns == "DateTime":
            continue
        elif "_target_" in columns:
            continue
        elif "_lag_" in columns:
            continue
        if (
            target_set[columns].dtype == "float32"
            or target_set[columns].dtype == "int64"
        ):
            target_set[columns + "_rolling_3d_sum"] = (
                target_set[columns].rolling(72, min_periods=1, center=False).sum()
            )
            target_set[columns + "_rolling_3d_min"] = (
                target_set[columns].rolling(72, min_periods=1, center=False).min()
            )
            target_set[columns + "_rolling_3d_max"] = (
                target_set[columns].rolling(72, min_periods=1, center=False).max()
            )
            target_set[columns + "_rolling_3d_mean"] = (
                target_set[columns].rolling(72, min_periods=1, center=False).mean()
            )
            target_set[columns + "_rolling_7d_sum"] = (
                target_set[columns].rolling(168, min_periods=1, center=False).sum()
            )
            target_set[columns + "_rolling_7d_min"] = (
                target_set[columns].rolling(168, min_periods=1, center=False).min()
            )
            target_set[columns + "_rolling_7d_max"] = (
                target_set[columns].rolling(168, min_periods=1, center=False).max()
            )
            target_set[columns + "_rolling_7d_mean"] = (
                target_set[columns].rolling(168, min_periods=1, center=False).mean()
            )
            target_set[columns + "_lag_1d"] = target_set[columns].shift(24)
            target_set[columns + "_lag_2d"] = target_set[columns].shift(48)
            target_set[columns + "_lag_3d"] = target_set[columns].shift(72)
            target_set[columns + "_lag_7d"] = target_set[columns].shift(168)
    target_set.ffill(inplace=True)
    target_set.fillna(-0.0000001, inplace=True)
    target_set.columns = target_set.columns.str.replace(" ", "_")
    return target_set


def train_test_periodic_split(result, percentage=0.8, step=0.1):
    """
    Пока параметр step лучше не менять +- корректно работает в режимах  до 0.1 включительно

    Функция создает обучаюшую и тестовую выборки

    in->:
    result - исходный набор данных, после всех предварительных обработок
    percentage - процентная составляющая обучающей выборки от общей
    step - шаг дискретизации
    out<-:
    train_set,test_set - обучающая и тестовая выборки
    """
    full_dt_len = result.DateTime.max() - result.DateTime.min()
    train_set = pd.DataFrame(columns=result.columns)
    test_set = pd.DataFrame(columns=result.columns)
    min_date = result.DateTime.min()
    start_date = full_dt_len / (10 / step)
    end_date = 2 * full_dt_len / (10 / step)
    counter = step
    while end_date <= full_dt_len:
        if counter <= percentage:
            train_set = pd.concat(
                [
                    train_set,
                    result.loc[
                        (result.DateTime >= min_date + start_date)
                        & (result.DateTime <= min_date + end_date)
                    ],
                ],
                ignore_index=True,
            )
            counter += step
        elif counter > percentage:
            test_set = pd.concat(
                [
                    test_set,
                    result.loc[
                        (result.DateTime >= min_date + start_date)
                        & (result.DateTime <= min_date + end_date)
                    ],
                ],
                ignore_index=True,
            )
            counter += step
        if counter >= 1:
            counter = 0.0
        start_date += full_dt_len / (10 / step)
        end_date += full_dt_len / (10 / step)
    return train_set, test_set


def assign_features_ks_hours(result, column_names, mode="Pin", ks="15", hours=48):
    """
    Функция берет n-признаков на основе имеющегося датасета модели  (в данном случае LGBMRegressor)
    """
    result.to_parquet("test.parquet")
    result_2 = result[column_names]
    result_2["DateTime"] = result["DateTime"]
    return result_2


def create_datasets():
    """
    Функция создания датасета
    Проходимся по Target_Name (названия КС в формате "КС-15"), вложенный цикл по сдвигам дней (48, 72, 96),
    вложенный цикл по направлению давления (входное/выходное - "Pin"/"Pout")

    """

    #! DEBUG  измени перед релизом
    for Target_Name in ["КС-15", "КС-16", "КС-17", "КС-19"]:
    # for Target_Name in ["КС-15"]:
        for Hours in [48, 72, 96]:
        # for Hours in [48]:
            Target_List = ["КС-15", "КС-16", "КС-17", "КС-19"]
            for mode in ["Pin", "Pout"]:
                try:
                # for mode in ["Pin"]:
                    # t_set = prepare_dataset(Target_List, Target_Name, "2",hours= Hours) #* В папке upload должны будут лежать

                    try:
                        t_set = pd.read_excel(
                            "uploaded/"
                            + Target_Name
                            + "_"
                            + mode
                            + "_"
                            + str(Hours)
                            + "_h.xlsx"
                        )
                    except Exception as e:
                        raise ("Read excel for target error:\n" + e)

                    # ! DEBUG
                    # try:
                    #     weather_dfs = prepare_weather_dict_dataset()

                    # except Exception as e:
                    #     raise("Prepare weather dict error:\n"+e)

                    try:
                        result = prepare_target_set(t_set, [])  #! DEBUG
                    except Exception as e:
                        raise ("Prepare target set error:\n" + e)

                    try:
                        total_result = prepare_shift_and_lags(result)
                    except Exception as e:
                        raise ("Prapare shift and lags error:\n" + e)

                    try:
                        total_result_new = total_result.rename(
                            columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x)
                        )
                        new_names = {
                            col: re.sub(r"[^A-Za-zА-Яа-я0-9_]+", "", col)
                            for col in total_result.columns
                        }
                        new_n_list = list(new_names.values())
                        # [LightGBM] Исправление одинаковых колонок.
                        new_names = {
                            col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
                            for i, (col, new_col) in enumerate(new_names.items())
                        }
                        total_result_new = total_result.rename(columns=new_names)
                    except Exception as e:
                        raise ("Rename columns error:\n" + e)

                    try:
                        train_set_column_names = (
                            pd.read_parquet(
                                "data/"
                                + Target_Name
                                + "_"
                                + mode
                                + "_"
                                + str(Hours)
                                + "_h.parquet"
                            )
                            .rename(
                                columns={
                                    f"{mode}_target_shift_{Hours}h_{Target_Name[-2:]}": f"{mode}_target_shift_{Hours}h_КС{Target_Name[-2:]}"
                                },
                            )
                            .columns
                        )
                    except Exception as e:
                        raise ("Error getting columns names from train dataset:\n" + e)

                    try:
                        result_final = assign_features_ks_hours(
                            total_result_new,
                            train_set_column_names,
                            mode,
                            Target_Name[-2:],
                            Hours,
                        )
                    except Exception as e:
                        raise ("Assign features ks hours error:\n" + e)

                    try:
                        result_final.to_parquet(
                            "data/"
                            + Target_Name
                            + "_"
                            + mode
                            + "_"
                            + str(Hours)
                            + "_h_new.parquet",
                            index=None,
                        )
                    except Exception as e:
                        raise ("Error saving result parquet file:\n" + e)
                except:
                    continue
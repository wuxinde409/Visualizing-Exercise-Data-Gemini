import openai
import json
import os
import re
import time
import pandas as pd
import gc
import psutil
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)  
pd.set_option('display.max_rows', None)  
import chardet
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import google.generativeai as genai
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from numpy.testing import assert_almost_equal
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Slider
from tabulate import tabulate
from sklearn.metrics import silhouette_score
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk,Spinbox
import threading
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from functools import partial
from tkinter import Scale, HORIZONTAL
from tkinter import Frame, Scale, HORIZONTAL, BOTTOM, TOP, X, BOTH
import math
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("環境變數 GEMINI_API_KEY 未設定（請加入.env 並設定自己的api key進去）。")
genai.configure(api_key=GEMINI_API_KEY)
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
# with open("./Metapunch Data/user_2023_9_6_16_28_49_.json","r",encoding="utf-8") as file:
#     data=json.load(file)
# print(data["summary"])

folder_path="./processed_users1/"
# folder_path="./Params/"
# 設定資料夾位置
json_columns=["totalPunchNum","maxPunchSpeed","avgReactionTime","hitRate","avgPunchSpeed","minReactionTime","maxPunchPower"]
boxing_df=pd.DataFrame(columns=json_columns)
last_filename = None 

def load_all_json_files():
    global boxing_df
    global max_values
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))  # 依據修改時間排序
    problematic_files = [] 
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        if os.path.getsize(file_path) > 0:  # 檢查檔案大小是否為 0
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    punch_power = data.get("punchPower", [])
                    max_punch_power = max(punch_power) if punch_power else None


                    min_reaction_time = data["summary"].get("minReactionTime")
                    avg_reaction_time = data["summary"].get("avgReactionTime")
                    if min_reaction_time == 3.5835 or avg_reaction_time == 3.5835:
                        problematic_files.append(file_path)
                
                    temp_row = {
                        "totalPunchNum": [data["summary"].get("totalPunchNum")],
                        "maxPunchSpeed": [data["summary"].get("maxPunchSpeed")],
                        "avgReactionTime": [data["summary"].get("avgReactionTime")],
                        "hitRate": [data["summary"].get("hitRate")],
                        "avgPunchSpeed": [data["summary"].get("avgPunchSpeed")],
                        "minReactionTime": [data["summary"].get("minReactionTime")],
                        "maxPunchPower": [max_punch_power]
                    }
                    
                    boxing_df = pd.concat([boxing_df, pd.DataFrame(temp_row)], ignore_index=True)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError in file '{filename}' at line {e.lineno}, column {e.colno}: {e.msg}")
                    print(f"File path: {file_path}")
                    # 當發生錯誤時，打印文件內容的前幾行來輔助調試
                    file.seek(0)  # 重置文件指針到開頭
                    print(f"File content preview:\n{file.read(20)}") 
    # 過濾資料
    over500power=boxing_df.loc[boxing_df['maxPunchPower']>500,'maxPunchPower'] #選擇特定行跟列用.loc比較方便
    # print(f"over500power:{over500power}")
    # print(f"len:{len(over500power)}")
    boxing_df=boxing_df[(boxing_df['totalPunchNum'] >= 0) & (boxing_df['maxPunchSpeed'] <=10) & (boxing_df['maxPunchSpeed'] >0) &(boxing_df['maxPunchPower'] <=1000)] 
    max_values=boxing_df[json_columns].apply(max)
    # max_values_list = boxing_df[json_columns].apply(max).tolist()
    print(f"max_values:\n{max_values}")

    # for file in problematic_files:
    #     print(file)


def get_latest_file(): 
    files=[]
    a=0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            files.append(filename)
            a+=1
    # if not files:#如果找不到文件就結束
    #     return None
    # latest_file=max(files, key=lambda f:os.path.getmtime(os.path.join(folder_path,filename)))
    # latest_file=max(files, key=lambda f:os.path.getmtime(os.path.join(folder_path,f)))


    # return (os.path.join(folder_path,latest_file))
    return(files[-1])


def check_if_updated(file_path):
    global last_summary
    with open (file_path,"r",encoding='utf-8') as f :
        data=json.load(f)
        current_summary=data.get("summary")
        if current_summary != last_summary:
            last_summary=current_summary
            return True
    return False


def check_percentage(new_row):
    global boxing_df
    means=boxing_df.mean(axis=0).astype(float)
    percentage={}
    for column in json_columns:
        current_value=new_row[column]
        percentage[column]= ((current_value*50)/means[column])
    print(percentage)
    top_two = sorted(percentage.items(), key=lambda item: item[1], reverse=True)[:2]
    print(f"Top two categories : {top_two}")

def plot_data(new_row):
    global boxing_df
    global max_values
    global json_columns
    fig, axes = plt.subplots(2, 4, figsize=(12, 8))  # 建立 2x4 圖片
    axes = axes.flatten()  # 讓它變成一維，便於迭代處理
    
    titles = [
        "Total Punches Number Analysis", "Max Punch Speed Analysis", 
        "Avg Reaction Time Analysis", "Hit Rate Analysis", 
        "Avg Punch Speed Analysis", "Min Reaction Time Analysis",
        "Max Punch Power Analysis"
    ]
    
    means = boxing_df.mean(axis=0).astype(float)
    print(f"Means value{means}")
    print(f"max_values:\n{max_values}")

    percent_dict = {}
    for i, column in enumerate(json_columns):
        if isinstance(new_row, pd.DataFrame):
            value = new_row.loc[0, column]  # 使用 .loc 
        else:
            value = new_row[column]  # 如果是 Series 或字典，直接使用

        # 使用 seaborn 畫出直方圖
        sns.histplot(boxing_df[column], ax=axes[i], bins=20, kde=True, color="blue")
        
        # percentdata = (100 - (value * 50 / means[column]))
        percentdata = abs( 100-(value * 100/ max_values[column]))
        axes[i].axvline(value, color="red", linestyle="--", label="Your Data")         
        percent_dict[column]=round(percentdata,2)


        # 使用 seaborn 畫出直方圖
        # hist = sns.histplot(boxing_df[column], ax=axes[i], bins=20, kde=True, color="blue")

        # percentdata = (means[column] * 50 // new_row[column])
        # percentdata = (100-(new_row[column] * 50 // means[column]))
        # axes[i].axvline(new_row[column], color="red", linestyle="--", label="Your Data")  # 突出顯示新數據
        
        
        axes[i].axvline(means[column], color="orange", linestyle="--", label="Mean Data")  # 平均數據
        
        title_font = {'fontsize': 11, 'fontweight': 'bold', 'color': 'darkblue'}
        axes[i].set_title(titles[i], pad=5, fontdict=title_font)  
        axes[i].set_xlabel(f"Top {percentdata:.2f}% of {column}", fontsize=9)
        axes[i].set_ylabel("Total participants")
        
        if titles[i] == "Hit Rate Analysis":  # 將圖例移動到左邊
            axes[i].legend(loc="upper left", bbox_to_anchor=(-0.5, 1), prop={'size': 8}, borderaxespad=0.)
        else:
            axes[i].legend(loc="upper right", bbox_to_anchor=(1, 1), prop={'size': 8}, borderaxespad=0.)
        
        # 調整圖例
        handles, labels = axes[i].get_legend_handles_labels()
        handles.append(plt.Rectangle((0, 0), 1, 1, fc="blue"))
        labels.append("All Data")
        axes[i].legend(handles=handles, labels=labels, loc="upper right", bbox_to_anchor=(1, 1), prop={'size': 8})
    percentage_series = pd.Series(percent_dict)
    print(f"percentage_series{percentage_series}")
    # 特殊處理最右下角的數據顯示圖
    i = 7  # 最右下角的圖表索引
    axes[i].axis('off')  
    axes[i].set_title("Player Data ", pad=5, fontdict=title_font)
    
    
    data_text = (
        f"{'totalPunchNum':<20} {new_row['totalPunchNum']:<10.4f}\n"
        f"{'maxPunchSpeed':<20} {new_row['maxPunchSpeed']:<10.4f}\n"
        f"{'avgReactionTime':<20} {new_row['avgReactionTime']:<10.4f}\n"
        f"{'hitRate':<20} {new_row['hitRate']:<10.4f}\n"
        f"{'avgPunchSpeed':<20} {new_row['avgPunchSpeed']:<10.4f}\n"
        f"{'minReactionTime':<20} {new_row['minReactionTime']:<10.4f}\n"
        f"{'maxPunchPower':<20} {new_row['maxPunchPower']:<10.4f}"
    )
    
    # 使用等寬字體（monospace）
    axes[i].text(
        0.05, 0.5, data_text, fontsize=10, 
        verticalalignment='center', horizontalalignment='left', 
        transform=axes[i].transAxes, fontfamily='monospace',bbox=dict(facecolor='none', edgecolor='black')
    )
    
    # 添加黑色邊框
    for spine in axes[i].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # 調整框線寬度
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    plt.tight_layout()  # 保持合適的間距
    # plt.show()  
    plt.savefig("seven_summing.png")
    plt.close()
    return percentage_series 





def process_new_file(file_path):#針對新加進來處理，並提取row來進行運算
    global boxing_df
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        punch_power = data.get("punchPower", [])
        max_punch_power = max(punch_power) if punch_power else None
    temp_row = {
        "totalPunchNum": data["summary"].get("totalPunchNum", [0])[0] if isinstance(data["summary"].get("totalPunchNum"), list) else data["summary"].get("totalPunchNum"),
        "maxPunchSpeed": data["summary"].get("maxPunchSpeed", [0])[0] if isinstance(data["summary"].get("maxPunchSpeed"), list) else data["summary"].get("maxPunchSpeed"),
        "avgReactionTime": data["summary"].get("avgReactionTime", [0])[0] if isinstance(data["summary"].get("avgReactionTime"), list) else data["summary"].get("avgReactionTime"),
        "hitRate": data["summary"].get("hitRate", [0])[0] if isinstance(data["summary"].get("hitRate"), list) else data["summary"].get("hitRate"),
        "avgPunchSpeed": data["summary"].get("avgPunchSpeed", [0])[0] if isinstance(data["summary"].get("avgPunchSpeed"), list) else data["summary"].get("avgPunchSpeed"),
        "minReactionTime": data["summary"].get("minReactionTime", [0])[0] if isinstance(data["summary"].get("minReactionTime"), list) else data["summary"].get("minReactionTime"),
        "maxPunchPower": max_punch_power
    }
    # temp_row = {
    #     "totalPunchNum": [data["summary"].get("totalPunchNum")],
    #     "maxPunchSpeed": [data["summary"].get("maxPunchSpeed")],
    #     "avgReactionTime": [data["summary"].get("avgReactionTime")],
    #     "hitRate": [data["summary"].get("hitRate")],
    #     "avgPunchSpeed": [data["summary"].get("avgPunchSpeed")],
    #     "minReactionTime": [data["summary"].get("minReactionTime")],
    #     "maxPunchPower": [max_punch_power]
    # }
    
    # 將新資料加入 dataframe
    # new_df = pd.DataFrame(temp_row)
    new_df = pd.DataFrame([temp_row])
    print(new_df)
    # check_percentage(new_df.iloc[0])
    boxing_df = pd.concat([boxing_df, new_df], ignore_index=True)

    boxing_df = boxing_df[(boxing_df['totalPunchNum'] >= 0) &
                          (boxing_df['maxPunchSpeed'] <= 10) & (boxing_df['maxPunchSpeed'] > 0)& (boxing_df['maxPunchPower'] < 1000)]
    plot_data(new_df.iloc[0])#這邊使用了plot_data func 來幫助生產六張視覺圖  
    print("新資料以處理完")
    print(plot_data(new_df.iloc[0]))
    
    
    
def find_optimal_clusters(data, max_clusters=10):
    # 先對資料進行標準化處理
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 存放每個 K 值下的 inertia 和 silhouette score
    inertia_list = []
    silhouette_list = []

    # Elbow Method: 迴圈從 1 開始
    for n_clusters in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
        kmeans.fit(data_scaled)
        inertia_list.append(kmeans.inertia_)  # 紀錄 inertia 值

        # Silhouette score 需要至少兩個簇
        if n_clusters > 1:
            silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
            silhouette_list.append(silhouette_avg)
    
    # 建立 1x2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 繪製 Elbow Method 圖表
    ax1.plot(range(1, max_clusters+1), inertia_list, marker='o', linestyle='--')
    ax1.set_title('Elbow Method for Optimal Clusters')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    ax1.set_xticks(range(1, max_clusters+1))

    # 繪製 silhouette score 圖表 (從 n_clusters=2 開始)
    ax2.plot(range(2, max_clusters+1), silhouette_list, marker='o', color='orange', linestyle='--')
    ax2.set_title('Silhouette Score for Optimal Clusters')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)
    ax2.set_xticks(range(2, max_clusters+1))

    plt.tight_layout()  # 自動調整子圖之間的間距
    plt.show()

def show_playerRHandDirLogs(file_path):
    with open(file_path,"r",encoding="utf-8") as file:
        data=json.load(file)
        playerRHandDirLogs=data.get("playerRHandDirLogs", [])
        x_coords=[]
        y_coords=[]
        z_coords=[]
        for point in playerRHandDirLogs:
            x_coords.append(point["x"])
            y_coords.append(point["y"])
            z_coords.append(point["z"])
        plt.scatter(x_coords,z_coords,color="red",marker='o')

        plt.show()
def test_show_playerRHandPosLogs(file_path): #顯示出2D的所有全集圖
    with open(file_path,"r",encoding="utf-8") as file:
        data=json.load(file)
        playerRHandDirLogs=data.get("playerRHandPosLogs", [])
        z_value=[point["z"]for point in playerRHandDirLogs]
        max_z=max(z_value)
        x_coords=[point["x"]for point in playerRHandDirLogs]
        y_coords=[point["y"]for point in playerRHandDirLogs]
        z_RHandPos_coords=[point["z"]for point in playerRHandDirLogs]
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # 计算每个点到中心点的角度
        angles = np.arctan2(y_coords - center_y, x_coords - center_x)
        
        # 创建颜色映射，基于角度对不同区域进行分类
        colors = []
        for angle in angles:
            if -np.pi/5 <= angle < np.pi/5:  # 右手
                colors.append('red')
            elif np.pi/5 <= angle < 3*np.pi/5:  # 頭部
                colors.append('blue')
            elif 3*np.pi/5 <= angle <= np.pi:  # 左手
                colors.append('yellow')
            elif -3*np.pi/5 <= angle < -np.pi/5:  # 右腳
                colors.append('green')
            elif -np.pi <= angle < -3*np.pi/5:  # 左腳
                colors.append('purple')

        # 繪製圖並分類
        plt.scatter(x_coords, y_coords, c=colors, marker='o')
        # plt.scatter(center_x, center_y, color="black", label="Center")  # 標記中心點
        plt.title("Analysis of the total number of punches in the hit regions")
        legend_elements = [
            Patch(facecolor='blue', edgecolor='black', label='Top'),
            Patch(facecolor='red', edgecolor='black', label='Right'),
            Patch(facecolor='yellow', edgecolor='black', label='Left'),
            Patch(facecolor='green', edgecolor='black', label='Bottom Right '),
            Patch(facecolor='purple', edgecolor='black', label='Bottom Left')
        ]
        plt.legend(handles=legend_elements, loc='upper left')  # 添加表格
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.show()
        
        
        # plt.scatter(x_RHandPos_coords,y_RHandPos_coords,color="blue")
        # plt.show()
    
        # 過濾 z 軸等於 0.175 的點
        
        # filtered_x = []
        # filtered_y = []
        # filtered_z = []
        
        # for point in playerRHandDirLogs:
        #     if (point["z"] - 0.175) > 0:  # 檢查 z 值是否接近 0.175
        #         filtered_x.append(point["x"])
        #         filtered_y.append(point["y"])
        #         filtered_z.append(point["z"])

        # # 如果有符合條件的點，繪製 2D 圖
        # if filtered_x and filtered_y:
        #     plt.scatter(filtered_x, filtered_y, color="blue", marker='o')
        #     plt.title("Filtered 2D Points with z=0.175")
        #     plt.xlabel("X Axis")
        #     plt.ylabel("Y Axis")
        #     plt.show()
        # else:
        #     print("No points found with z=0.175")


def show_playerRHandLHandPosLogs(file_path): #顯示出2D的所有拳擊圖
    with open(file_path,"r",encoding="utf-8") as file:
        data=json.load(file)
        PlayerRHandPosLogs=data.get("playerRHandPosLogs", [])
        PlayerLHandPosLogs=data.get("playerLHandPosLogs", [])
        
        x_RHandPos_coords=[point["x"]for point in PlayerRHandPosLogs]
        y_RHandPos_coords=[point["y"]for point in PlayerRHandPosLogs]

        x_LHandPos_coords=[point["x"]for point in PlayerLHandPosLogs]
        y_LHandPos_coords=[point["y"]for point in PlayerLHandPosLogs]

    
        plt.figure()#在這裡清乾淨畫布 
       
        plt.scatter(x_RHandPos_coords,y_RHandPos_coords, color="red" ,s=5,marker='o',label="Right Hand")
        plt.scatter(x_LHandPos_coords,y_LHandPos_coords, color="blue",s=5 ,marker='o',label="Left Hand")
        # plt.scatter(center_x, center_y, color="black", label="Center")  # 標記中心點
        plt.title("Analysis of the different hands of punches in the hit regions")
        plt.legend(loc="upper right")
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        # plt.show()
        plt.savefig("Analysis of the different hands of punches in the hit regions.png")
        plt.close()

def show_playerRHandPosLogs(file_path):#有處理平均值的
    with open(file_path,"r",encoding="utf-8") as file:
        data=json.load(file)
        playerRHandPosLogs=data.get("playerRHandPosLogs", [])
        avg_playerRHandPosLogs=[]
        
        for i in range(0,len(playerRHandPosLogs),30):
            each=playerRHandPosLogs[i:i+30]
            avg_x = round(np.mean([point["x"] for point in each]), 4)
            avg_y = round(np.mean([point["y"] for point in each]), 4)  
            avg_z = round(np.mean([point["z"] for point in each]), 4)
            avg_playerRHandPosLogs.append({"x":avg_x,"y":avg_y,"z":avg_z}) 
        x_RHandPos_coords=[point["x"]for point in avg_playerRHandPosLogs]
        y_RHandPos_coords=[point["y"]for point in avg_playerRHandPosLogs]
        z_RHandPos_coords=[point["z"]for point in avg_playerRHandPosLogs]
        
        bagPosLogs=data.get("bagPosLogs",[])
        avg_bagPosLogs=[]
        for i in range(0,len(bagPosLogs),30):
            each=bagPosLogs[i:i+30]
            avg_x=round(np.mean([point["x"] for point in each]),4)
            avg_y=round(np.mean([point["y"] for point in each]),4)
            avg_z=round(np.mean([point["z"] for point in each]),4)
            avg_bagPosLogs.append({"x":avg_x,"y":avg_y,"z":avg_z})
        x_bagPos_coords=[point["x"] for point in avg_bagPosLogs]
        y_bagPos_coords=[point["y"] for point in avg_bagPosLogs]
        z_bagPos_coords=[point["z"] for point in avg_bagPosLogs]

        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        # ax.plot(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",marker='o',markersize=3)
        # ax.plot(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",marker='o',markersize=3)
        ax.scatter(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",s=0.75)
        ax.scatter(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",s=0.75)
        ax.set_title("The position of the hand and the bag")
        ax.set_xlim(18.5,19.5)
        # ax.set_ylim(0.15,0.2)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Z Axis')                                                                                 
        ax.set_zlabel('Y Axis')
        ax.view_init(elev=31,azim=-99)#拿來調整最後圖的旋轉
        plt.show()           
         
def show_2D_playerRHandPosLogs(file_path):
    with open(file_path,"r",encoding="utf-8") as file:
        data=json.load(file)
        playerRHandPosLogs=data.get("playerRHandPosLogs", [])
        avg_playerRHandPosLogs=[]
        
        for i in range(0,len(playerRHandPosLogs),30):
            each=playerRHandPosLogs[i:i+30]
            avg_x = round(np.mean([point["x"] for point in each]), 4)
            avg_y = round(np.mean([point["y"] for point in each]), 4)  
            avg_z = round(np.mean([point["z"] for point in each]), 4)
            avg_playerRHandPosLogs.append({"x":avg_x,"y":avg_y,"z":avg_z}) 
        x_RHandPos_coords=[point["x"]for point in avg_playerRHandPosLogs]
        y_RHandPos_coords=[point["y"]for point in avg_playerRHandPosLogs]
        z_RHandPos_coords=[point["z"]for point in avg_playerRHandPosLogs]
        
        bagPosLogs=data.get("bagPosLogs",[])
        avg_bagPosLogs=[]
        for i in range(0,len(bagPosLogs),30):
            each=bagPosLogs[i:i+30]
            avg_x=round(np.mean([point["x"] for point in each]),4)
            avg_y=round(np.mean([point["y"] for point in each]),4)
            avg_z=round(np.mean([point["z"] for point in each]),4)
            avg_bagPosLogs.append({"x":avg_x,"y":avg_y,"z":avg_z})
        x_bagPos_coords=[point["x"] for point in avg_bagPosLogs]
        y_bagPos_coords=[point["y"] for point in avg_bagPosLogs]
        z_bagPos_coords=[point["z"] for point in avg_bagPosLogs]
    
        fig,ax=plt.subplots() 
        
        ax.scatter(x_RHandPos_coords,y_RHandPos_coords,color="blue",s=2)
        ax.scatter(x_bagPos_coords,y_bagPos_coords,color="red",s=2)
        ax.set_title("The position of the hand and the bag")
        ax.set_xlim(18.5,19.5)
        # ax.set_ylim(0.15,0.2)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Z Axis')                                                                                 
        # ax.set_zlabel('Y Axis')
        # ax.view_init(elev=31,azim=-99)#拿來調整最後圖的旋轉
        plt.show()                

        
def animate_playerRHand_LHandPosLogs(file_path,frame_limit):
    with open (file_path,"r",encoding="utf-8")as file:
        data=json.load(file)
        playerRHandPosLogs = data.get("playerRHandPosLogs", [])
        playerLHandPosLogs = data.get("playerLHandPosLogs", [])
        bagPosLogs = data.get("bagPosLogs", [])
        
        x_RHandPos_coords = [point["x"] for point in playerRHandPosLogs]
        y_RHandPos_coords = [point["y"] for point in playerRHandPosLogs]
        z_RHandPos_coords = [point["z"] for point in playerRHandPosLogs]

        x_LHandPos_coords = [point["x"] for point in playerLHandPosLogs]
        y_LHandPos_coords = [point["y"] for point in playerLHandPosLogs]
        z_LHandPos_coords = [point["z"] for point in playerLHandPosLogs]

        x_bagPos_coords = [point["x"] for point in bagPosLogs]
        y_bagPos_coords = [point["y"] for point in bagPosLogs]
        z_bagPos_coords = [point["z"] for point in bagPosLogs]

        # 創建圖表
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter_r = ax.scatter([], [], [], color="blue", s=8, label="Right hand")
        scatter_l = ax.scatter([], [], [], color="green", s=8, label="Left hand")
        scatter_bag = ax.scatter(x_bagPos_coords, z_bagPos_coords, y_bagPos_coords, color="red", s=8, label="Bag")

        ax.set_xlim(min(x_LHandPos_coords), max(x_RHandPos_coords))
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Z Axis')
        ax.set_zlabel('Y Axis')
        # ax.set_ylim(-2.8,-2)
        ax.set_ylim((min(min(z_RHandPos_coords),min(z_LHandPos_coords))),max(max(z_RHandPos_coords),max(z_LHandPos_coords)))
        ax.set_title("The position of the both hands and the bag")
        ax.view_init(elev=37, azim=-64)
        ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1), borderaxespad=0.)

        def update(frame):
            # 計算當前顯示的點範圍
            
            if frame_limit == "all":
                current_index= frame+1
                start_index=0
            else:
                current_index = min(len(x_RHandPos_coords), frame + frame_limit)#現在要去的點
                start_index=current_index-frame_limit
            # start_index = max(0, current_index - frame_limit)

            # 更新右手、左手的點
            scatter_r._offsets3d = (x_RHandPos_coords[start_index:current_index],
                                     z_RHandPos_coords[start_index:current_index],
                                     y_RHandPos_coords[start_index:current_index])

            scatter_l._offsets3d = (x_LHandPos_coords[start_index:current_index],
                                     z_LHandPos_coords[start_index:current_index],
                                     y_LHandPos_coords[start_index:current_index])

            return scatter_r, scatter_l
        
        # 動畫
        ani = FuncAnimation(fig, update, frames=np.arange(0, len(x_RHandPos_coords)), interval=100, blit=False) #interval 用來調整速度的
        
        plt.show()
                                                                                                                                                                                                                                      
# def show_playerRHandPDirLogs(file_path):
#     with open(file_path,"r",encoding="utf-8") as file:
#         data=json.load(file)
#         playerRHandDirLogs=data.get("playerRHandDirLogs", [])
#         x_RHandDir_coords=[point["x"]for point in playerRHandDirLogs]
#         y_RHandDir_coords=[point["y"]for point in playerRHandDirLogs]
#         z_RHandDir_coords=[point["z"]for point in playerRHandDirLogs]
        
#         bagPosLogs=data.get("bagPosLogs",[])
#         x_bagPos_coords=[point["x"] for point in bagPosLogs]
#         y_bagPos_coords=[point["y"] for point in bagPosLogs]
#         z_bagPos_coords=[point["z"] for point in bagPosLogs]

        
              
#         fig=plt.figure()
#         ax=fig.add_subplot(111,projection='3d')
#         # ax.plot(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",marker='o',markersize=3)
#         # ax.plot(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",marker='o',markersize=3)
#         ax.scatter(x_RHandDir_coords,z_RHandDir_coords,y_RHandDir_coords,color="blue",s=0.75)
#         ax.scatter(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",s=0.75)
#         ax.set_title("The position of the hand and the bag")
#         # ax.set_ylim(0.15,0.2)
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Z Axis')                                                                                 
#         ax.set_zlabel('Y Axis')
#         ax.view_init(elev=31,azim=-99)#拿來調整最後圖的旋轉
#         plt.show()     
        
def show_playerPosLogs(file_path):
    with open(file_path,"r",encoding="utf-8") as file:              
        data=json.load(file)                                                                                                
        playerPosLogs=data.get("playerPosLogs", [])                                                                                                                                                 
        x_coords=[]
        y_coords=[]
        z_coords=[]
        for point in playerPosLogs:
            x_coords.append(point["x"])
            y_coords.append(point["y"])
            z_coords.append(point["z"])
        # plt.scatter(x_coords,z_coords,color="blue",marker='o')
        # plt.plot(x_coords,z_coords,color="blue",marker='o')
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        plt.plot(x_coords,y_coords,z_coords,marker='o',markersize=2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))  # 调整间隔来减少轴标签的密集程度

        plt.title("playerPosLogs")
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.show()

       
def animate_skeleton_3D(file_path):
    # 讀取 JSON 檔案
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    playerRHandPosLogs = data.get("playerRHandPosLogs", [])

    # 提取 x, y, z 座標
    x_coords = [point["x"] for point in playerRHandPosLogs]
    y_coords = [point["y"] for point in playerRHandPosLogs]
    z_coords = [point["z"] for point in playerRHandPosLogs]

    # 創建三維圖
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 設置初始的點
    points, = ax.plot(x_coords, y_coords, z_coords, 'bo')

    # 更新骨架動作的函數
    def update(num):
        # 這裡的 num 代表第幾幀
        points.set_data(x_coords[:num], y_coords[:num])
        points.set_3d_properties(z_coords[:num])

    # 創建動畫
    ani = animation.FuncAnimation(fig, update, frames=len(x_coords), interval=100)

    plt.show()


def show_reactionTime(folder_path):
    with open(folder_path,'r',encoding="utf-8") as file:
        data=json.load(file)
    reaction_data=data.get('reactionTime',[])
    fig, ax = plt.subplots()
    ax.set_ylim(0, max(reaction_data) + 1)  # Y軸範圍
    x_positions = [i * 1.5 for i in range(len(reaction_data))]
    bar_container = ax.bar(x_positions, [0] * len(reaction_data),width=1.5,color='orange',edgecolor='black')  # 创建空的条形图

    # 更新每一貞數據
    def update(frame):
        for i, bar in enumerate(bar_container):
            bar.set_height(reaction_data[i] if i <= frame else 0)  # 根據每個貞調整高度

    # 創動畫
    ani = animation.FuncAnimation(fig, update, frames=len(reaction_data), interval=200, repeat=False)

    # 顯示動畫
    plt.title("ReactionTime per punch ")
    plt.ylabel("ReactionTime")
    plt.xlabel("Total punch number")
    plt.show()

def show_player_and_bag_pos_logs(file_path,img_path):#有bag背景圖
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
        playerPosLogs = data.get("playerPosLogs", [])
        x_coords_player = [point["x"] for point in playerPosLogs]
        y_coords_player = [point["y"] for point in playerPosLogs]
        z_coords_player = [point["z"] for point in playerPosLogs]
        
        bagPosLogs = data.get("bagPosLogs", [])
        x_coords_bag = [point["x"] for point in bagPosLogs]
        y_coords_bag = [point["y"] for point in bagPosLogs]
        z_coords_bag = [point["z"] for point in bagPosLogs]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='.', color='blue', label=' Head Position')   
        ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='.', color='red', label='Bag Position')
        
        # ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='o', markersize=2, color='blue', label='Player Position')   
        # ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='o', markersize=2, color='red', label='Bag Position')
        
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
        plt.title("The absolute position of the Head and the Bag")
        # ax.set_xlabel('X Axis')
        ax.set_xlabel('Z Axis')
        ax.set_ylabel('X Axis')
        ax.set_zlabel('Y Axis')
        ax.legend(loc='upper left')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), borderaxespad=0.)
        # ax.invert_xaxis()
        # ax.invert_zaxis()#反轉Z軸數字
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))#x軸
        ax.zaxis.set_major_locator(ticker.MultipleLocator(0.5))#y軸
        img = plt.imread(img_path)  # 讀取圖片
        # 获取 X 和 Y 的边界，用来设置图片的范围
        xlim = [min(x_coords_bag) - 0.1, max(x_coords_bag) + 6]
        ylim = [min(y_coords_bag) - 0.1, max(y_coords_bag) + 5]
        img_extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        img_ax = fig.add_axes([0.3, 0.35, 0.4, 0.4], zorder=0)
        img_ax.imshow(img, extent=img_extent, alpha=0.2)
        img_ax.set_xlim(xlim)
        img_ax.set_ylim(ylim)
        img_ax.axis('off')  # 隐藏坐标轴
        # ax.set_xlim(19.2,19.5) #直接設定X怎樣
        ax.view_init(elev=22, azim=176)  
        plt.show()    


        
def jump_show_head_and_bag_pos_logs(file_path):#使用者head位置與沙包位置3d
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
        playerPosLogs = data.get("playerPosLogs", [])
        x_coords_player = [point["x"] for point in playerPosLogs]
        y_coords_player = [point["y"] for point in playerPosLogs]
        z_coords_player = [point["z"] for point in playerPosLogs]
        
        bagPosLogs = data.get("bagPosLogs", [])
        x_coords_bag = [point["x"] for point in bagPosLogs]
        y_coords_bag = [point["y"] for point in bagPosLogs]
        z_coords_bag = [point["z"] for point in bagPosLogs]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='.', color='blue', label=' Head Position')   
        ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='.', color='red', label='Bag Position')
        ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='o', markersize=2, color='blue', label='Player Position')   
        # ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='o', markersize=2, color='red', label='Bag Position')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        plt.title("The absolute position of the Head and the Bag")
        ax.set_xlabel('Z Axis')
        ax.set_ylabel('X Axis')
        ax.set_zlabel('Y Axis')
        # ax.set_xlim(19.3,19.8)#這裡設定X值會出錯要再研究
        ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1), borderaxespad=0.)
        ax.view_init(elev=62, azim=150)  
        plt.show()    


def show_twohands_and_bag_pos_logs(file_path,img_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
        playerPosLogs = data.get("playerPosLogs", [])
        x_coords_player = [point["x"] for point in playerPosLogs]
        y_coords_player = [point["y"] for point in playerPosLogs]
        z_coords_player = [point["z"] for point in playerPosLogs]
        
        bagPosLogs = data.get("bagPosLogs", [])
        x_coords_bag = [point["x"] for point in bagPosLogs]
        y_coords_bag = [point["y"] for point in bagPosLogs]
        z_coords_bag = [point["z"] for point in bagPosLogs]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='.', color='blue', label=' Head Position')   
        ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='.', color='red', label='Bag Position')
        
        # ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='o', markersize=2, color='blue', label='Player Position')   
        # ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='o', markersize=2, color='red', label='Bag Position')
        
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
        plt.title("The absolute position of the Head and the Bag")
        # ax.set_xlabel('X Axis')
        ax.set_xlabel('Z Axis')
        ax.set_ylabel('X Axis')
        ax.set_zlabel('Y Axis')
        ax.legend(loc='upper left')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), borderaxespad=0.)
        
        # ax.invert_xaxis()
        # ax.invert_zaxis()#反轉Z軸數字
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))#x軸
        ax.zaxis.set_major_locator(ticker.MultipleLocator(0.5))#y軸
        
        img = plt.imread(img_path)  # 讀取圖片
        
        xlim = [min(x_coords_bag) - 0.1, max(x_coords_bag) + 6]
        ylim = [min(y_coords_bag) - 0.1, max(y_coords_bag) + 5]
        img_extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        img_ax = fig.add_axes([0.3, 0.35, 0.4, 0.4], zorder=0)
        img_ax.imshow(img, extent=img_extent, alpha=0.2)

        img_ax.set_xlim(xlim)
        img_ax.set_ylim(ylim)
        img_ax.axis('off') 
        # ax.set_xlim(19.2,19.5) #直接設定X怎樣
        ax.view_init(elev=22, azim=176)  
        plt.show()
    
    

    # def update(frame):
    #     for i, bar in enumerate(bar_container1):
    #         bar.set_height(speed_data[i] if i<=frame else 0)
    #     for i, bar in enumerate(bar_container2):
    #         bar.set_height(reaction_data[i] if i<=frame else 0)
            
    # ani=animation.FuncAnimation(fig,update,frames=max(len(speed_data), len(reaction_data)), interval=200, repeat=False)
    # speed_legend_element=[Patch(facecolor='#00FFFF',edgecolor='black',label='First Half of Punch'),Patch(facecolor='blue',edgecolor='black',label='Seconf Half of Punch')]
    # reaction_legend_element=[Patch(facecolor='#FFFF37',edgecolor='black',label='First Half of Punch'),Patch(facecolor='orange',edgecolor='black',label='Second Half of Punch')]
    # ax1.legend(handles=speed_legend_element,loc='upper right')
    # ax2.legend(handles=reaction_legend_element,loc='upper right')
    # plt.tight_layout()
    # ani_variable = ani
    # plt.show()

    
  
def show_punch_position_2D(file_path):    
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        playerRHandPosLogs = data.get("playerRHandPosLogs", [])
        playerLHandPosLogs = data.get("playerLHandPosLogs", [])
        puncherIdx = data.get("puncherIdx", [])
        punchTimeCode = data.get("punchTimeCode", [])
        sec_num=len(playerRHandPosLogs)//30 
        rhand,lhand,r_xavg,r_yavg,r_zavg,r_number=[],[],[],[],[],[]
        for i in range(len(puncherIdx)) :
            if puncherIdx[i]==0:
                rhand.append(i) 
            else:
                lhand.append(i)
        for i in rhand:
            temp=playerRHandPosLogs[:round(punchTimeCode[i]*sec_num)]
            r_number.append(len(temp))
            x_sum=sum(point["x"] for point in temp)
            y_sum=sum(point["y"] for point in temp)
            z_sum=sum(point["z"] for point in temp)
            
            r_xavg.append(round(x_sum/len(temp),3))
            r_yavg.append(round(y_sum/len(temp),3))
            r_zavg.append(round(z_sum/len(temp),3))
            
        l_xavg,l_yavg,l_zavg,l_number=[],[],[],[]
        for i in lhand:
            temp=playerLHandPosLogs[:round(punchTimeCode[i]*sec_num)]
            # print(f"punchtimenum{punchTimeCode[i]}")
            l_number.append(len(temp))
            x_sum=sum(point["x"] for point in temp)
            y_sum=sum(point["y"] for point in temp)
            z_sum=sum(point["z"] for point in temp)
            
            l_xavg.append(round(x_sum/len(temp),3))
            l_yavg.append(round(y_sum/len(temp),3))
            l_zavg.append(round(z_sum/len(temp),3))
        bagPosLogs=data.get("bagPosLogs",[])
        x_bagPos_coords=[point["x"] for point in bagPosLogs]
        y_bagPos_coords=[point["y"] for point in bagPosLogs]
        z_bagPos_coords=[point["z"] for point in bagPosLogs]
        fig,ax=plt.subplots()

        # ax.plot(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",marker='o',markersize=3)
        # ax.plot(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",marker='o',markersize=3)
        ax.scatter(r_xavg,r_zavg,color="blue",s=4,label="Right hand")
        ax.scatter(l_xavg,l_zavg,color="green",s=4,label="Left hand")
        ax.scatter(x_bagPos_coords,z_bagPos_coords,color="red",s=4,label="Bag")
        ax.set_title("Punch position point per second")
        # ax.set_xlim(min(x_LHandPos_coords),max(x_RHandPos_coords))
        ax.set_xlim(min(min(r_xavg,),min(l_xavg,)),max(max(r_xavg,),max(l_xavg,),max(x_bagPos_coords)))
        # ax.set_ylim(0.15,0.2)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Z Axis')                                                                                 
        # ax.set_zlabel('Y Axis')
        ax.legend(loc='upper left', bbox_to_anchor=(0.83, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,fontsize='small')

        # plt.savefig("Punch position.png")
        plt.show()


# show_punch_position_2D("./processed_users1/user_2024_10_12_16_21_31_.json")  
       
   
   
   
   
   
def initialize_guardposition_frame(self):
    if hasattr(self,"guardposition_frame") and self.guardposition_frame is not None:
        return
    self.guardposition_frame = tk.Frame(self.canvas)
    
    self.frame_righthand_label = tk.Label(self.guardposition_frame,text="Move XYZ",font=("Arial",16, "bold"))
    self.frame_righthand_label.grid(row=0,column=0, padx=1,pady=1)
    self.frame_righthand_Stability_label = tk.Label(self.guardposition_frame,text="Stability",font=("Arial",16, "bold"))
    self.frame_righthand_Stability_label.grid(row=0,column=1, padx=1,pady=1)  

    self.frame_lefthand_label = tk.Label(self.guardposition_frame,text="Move XYZ",font=("Arial",16, "bold"))
    self.frame_lefthand_label.grid(row=0,column=2, padx=1,pady=1)
    self.frame_lefthand_Stability_label = tk.Label(self.guardposition_frame,text="Stability",font=("Arial",16, "bold"))
    self.frame_lefthand_Stability_label.grid(row=0,column=3, padx=1,pady=1)  
   
    self.punchnumber_spinbox_var = tk.StringVar(value="1")
    self.punchnumber_spinbox= tk.Spinbox(self.guardposition_frame,from_=1, to=100 ,textvariable=self.punchnumber_spinbox_var,font=("Arial",16, "bold"),width=5)
    self.punchnumber_spinbox.grid(row=0,column=4, padx=1,pady=1)  
    
def show_boxingstance(file_path):    #
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        playerRHandPosLogs = data.get("playerRHandPosLogs", [])
        playerLHandPosLogs = data.get("playerLHandPosLogs", [])
        playerPosLogs = data.get("playerPosLogs",[])
        puncherIdx = data.get("puncherIdx", [])
        punchTimeCode = data.get("punchTimeCode", [])
        reactionTime = data.get("reactionTime", [])
        totaltime=math.ceil(punchTimeCode[-1])
        print(f"totaltime{totaltime}")
        sec_num=len(playerRHandPosLogs)//totaltime
        print(f"sec_num{sec_num}")
        
        bagPosLogs=data.get("bagPosLogs",[])
        x_bagPos_coords=[point["x"] for point in bagPosLogs]
        y_bagPos_coords=[point["y"] for point in bagPosLogs]
        z_bagPos_coords=[point["z"] for point in bagPosLogs]
        bx=(x_bagPos_coords[0])
        bz=(z_bagPos_coords[0])
        
        initial_position=round(sec_num*10)
        
        
        def sum_array(array,initial_position):
            tempavg,xavg,yavg,zavg=[],[],[],[]
            temp=array[:initial_position]
            print(f"temp:5{temp[:5]}")
            tempavg.append(len(temp))
            x_sum=sum(point["x"] for point in temp)
            y_sum=sum(point["y"] for point in temp)
            z_sum=sum(point["z"] for point in temp)
            xavg.append(round(x_sum/len(temp),3))
            yavg.append(round(y_sum/len(temp),3))
            zavg.append(round(z_sum/len(temp),3))
            return xavg,yavg,zavg
        r_xavg,r_yavg,r_zavg=sum_array(playerRHandPosLogs,initial_position)
        l_xavg,l_yavg,l_zavg=sum_array(playerLHandPosLogs,initial_position)
        player_xavg,player_yavg,player_zavg=sum_array(playerPosLogs,initial_position)
        print(f"r_xavg: {r_xavg}")
        print(f"r_yavg: {r_yavg}")
        print(f"r_zavg: {r_zavg}")
        
        rhand,lhand=[],[]
        for i in range(len(puncherIdx)) :
            if puncherIdx[i]==0:
                rhand.append(i) 
            else:
                lhand.append(i)
        print(f"rhand{rhand}")
        print(f"lhand{lhand}")
        def calculate_offset(logs, start_time, end_time, sec_num, avg_x, avg_y, avg_z, avg_count=10):
            # 計算篩選區間內的數據點
            start_idx = round(start_time * sec_num)
            end_idx = round(end_time * sec_num)
            interval_points = logs[start_idx:end_idx]

            # 初始化最小距離和臨界點索引
            min_distance = float("inf")
            critical_idx = None

            for i in range(len(interval_points) - 1):
                x, y, z = interval_points[i]["x"], interval_points[i]["y"], interval_points[i]["z"]
                distance = ((x - avg_x) ** 2 + (y - avg_y) ** 2 + (z - avg_z) ** 2) ** 0.5

                next_x, next_y, next_z = (
                    interval_points[i + 1]["x"],
                    interval_points[i + 1]["y"],
                    interval_points[i + 1]["z"],
                )
                next_distance = ((next_x - avg_x) ** 2 + (next_y - avg_y) ** 2 + (next_z - avg_z) ** 2) ** 0.5

                # 找到最接近基準點且下一個點開始遠離的臨界點
                if distance < min_distance and next_distance > distance:
                    min_distance = distance
                    critical_idx = i

            if critical_idx is not None:
                # 向前取 avg_count 個點
                start_avg_idx = max(0, critical_idx - avg_count + 1)
                avg_points = interval_points[start_avg_idx : critical_idx + 1]
            else:
                # 如果找不到臨界點，使用最後的 avg_count 個點
                avg_points = interval_points[-avg_count:] if len(interval_points) >= avg_count else interval_points

            # 計算平均值
            x_avg = sum(point["x"] for point in avg_points) / len(avg_points)
            y_avg = sum(point["y"] for point in avg_points) / len(avg_points)
            z_avg = sum(point["z"] for point in avg_points) / len(avg_points)

            # 偏移量計算
            offset_x = round(x_avg - avg_x, 3)
            offset_y = round(y_avg - avg_y, 3)
            offset_z = round(z_avg - avg_z, 3)
            offset_distance = round((offset_x**2 + offset_y**2 + offset_z**2) ** 0.5, 3)

            # 偏移率（相對於基準點）
            base_distance = (avg_x**2 + avg_y**2 + avg_z**2) ** 0.5
            offset_rate = 100-(round((offset_distance / base_distance) * 100, 2)) if base_distance != 0 else 0

            return offset_x, offset_y, offset_z, offset_distance, offset_rate

        r_offset_data = []  # 儲存右手偏移量
        rl_offset_data = []  # 儲存右手與左手偏移量
        l_offset_data = []  # 儲存左手偏移量
        lr_offset_data = []  # 儲存左手與右手偏移量

        avg_rpositions_x = []  # 儲存每拳右手的 avg_x + offset_x
        avg_rpositions_y = []  # 儲存每拳右手的 avg_y + offset_y
        avg_rpositions_z = []  # 儲存每拳右手的 avg_z + offset_z

        avg_lpositions_x = []  # 儲存每拳左手的 avg_x + rl_offset_x
        avg_lpositions_y = []  # 儲存每拳左手的 avg_y + rl_offset_y
        avg_lpositions_z = []  # 儲存每拳左手的 avg_z + rl_offset_z

        for idx, j in enumerate(puncherIdx):  # 依序處理每拳
            if j == 0:  # 右手
                start_time = punchTimeCode[idx]
                end_time = totaltime if idx+1>= len(punchTimeCode) else punchTimeCode[idx+1]

                r_offset = calculate_offset(
                    playerRHandPosLogs,
                    start_time,
                    end_time,
                    sec_num,
                    r_xavg[0],
                    r_yavg[0],
                    r_zavg[0],
                    avg_count=10
                )
                rl_offset = calculate_offset(
                    playerLHandPosLogs,
                    start_time,
                    end_time,
                    sec_num,
                    l_xavg[0],
                    l_yavg[0],
                    l_zavg[0],
                    avg_count=10
                )
                r_offset_data.append(r_offset)
                rl_offset_data.append(rl_offset)

                # 加入右手偏移後的座標
                avg_rpositions_x.append(r_xavg[0] + r_offset[0])
                avg_rpositions_y.append(r_yavg[0] + r_offset[1])
                avg_rpositions_z.append(r_zavg[0] + r_offset[2])

                # 加入左手偏移後的座標
                avg_lpositions_x.append(l_xavg[0] + rl_offset[0])
                avg_lpositions_y.append(l_yavg[0] + rl_offset[1])
                avg_lpositions_z.append(l_zavg[0] + rl_offset[2])

            else:  # 左手
                start_time = punchTimeCode[idx]
                end_time = totaltime if idx+1>= len(punchTimeCode) else punchTimeCode[idx+1]

                l_offset = calculate_offset(
                    playerLHandPosLogs,
                    start_time,
                    end_time,
                    sec_num,
                    l_xavg[0],
                    l_yavg[0],
                    l_zavg[0],
                    avg_count=10
                )
                lr_offset = calculate_offset(
                    playerRHandPosLogs,
                    start_time,
                    end_time,
                    sec_num,
                    r_xavg[0],
                    r_yavg[0],
                    r_zavg[0],
                    avg_count=10
                )
                l_offset_data.append(l_offset)
                lr_offset_data.append(lr_offset)

                # 加入左手偏移後的座標
                avg_lpositions_x.append(l_xavg[0] + l_offset[0])
                avg_lpositions_y.append(l_yavg[0] + l_offset[1])
                avg_lpositions_z.append(l_zavg[0] + l_offset[2])

                # 加入右手偏移後的座標
                avg_rpositions_x.append(r_xavg[0] + lr_offset[0])
                avg_rpositions_y.append(r_yavg[0] + lr_offset[1])
                avg_rpositions_z.append(r_zavg[0] + lr_offset[2])
            
           
        print(f"avg_lpositions_x.{avg_lpositions_x},len(l)={len(avg_lpositions_x)}")
        print(f"avg_rpositions_x,{avg_rpositions_x},len(r)={len(avg_rpositions_x)}")       
           
           
            
        print("右手偏移量:")
        for i, (offset_x, offset_y, offset_z, distance, rate) in enumerate(r_offset_data):
            print(f"The {i+1} Punch: Move XYZ ({offset_x}, {offset_y}, {offset_z}), Stability: {rate}%")

        print("\n左手偏移量:")
        for i, (offset_x, offset_y, offset_z, distance, rate) in enumerate(l_offset_data):
            print(f"The  {i+1} Punch: Move  XYZ ({offset_x}, {offset_y}, {offset_z}), Stability: {rate}%")
            
            
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        # fig,ax=plt.subplots()

        ax.scatter(r_xavg,r_zavg,r_yavg,color="blue",s=15,label="Right hand")
        ax.scatter(l_xavg,l_zavg,l_yavg,color="green",s=15,label="Left hand")
        
        ax.scatter(avg_rpositions_x,avg_rpositions_z,avg_rpositions_y,color="darkturquoise",s=15,label="right hand point")
        ax.scatter(avg_lpositions_x,avg_lpositions_z,avg_lpositions_y,color="limegreen",s=15,label="left hand point")
        
        ax.scatter(player_xavg,player_zavg,player_yavg,color="purple",s=10,label="Head")
        ax.scatter(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",s=15,label="Bag")
        ax.set_title("Boxing Stance ")
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Z Axis')                                                                                 
        ax.set_zlabel('Y Axis')
        ax.legend(loc='upper left', bbox_to_anchor=(0.83, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,fontsize='small')
        # ax.view_init(elev=37,azim=-64)#拿來調整最後圖的旋轉
        plt.savefig("Punch position.png")
        plt.show()

    
# def start_monitoring():
#     load_all_json_files()
#     print(boxing_df.mean(axis=0))
#     global last_filename
#     last_filename=get_latest_file()
#     print(last_filename)
    
#     load_all_json_files()
#     while True:
#         latest_filename=get_latest_file()
#         if latest_filename and latest_filename!=last_filename:
#             last_filename=latest_filename
#             latest_file_path = os.path.join(folder_path, latest_filename)
#             print("發現新檔案")
#             process_new_file(latest_file_path)
#         else:
#             print("沒有新檔案")

#         time.sleep(5) 


# if __name__ == "__main__":
#     start_monitoring()

load_all_json_files()
# print(boxing_df.max())
means=boxing_df.mean(axis=0).astype(float)
# print(f"means={means}")







# find_optimal_clusters(boxing_df, max_clusters=10)#找到適合分成哪幾種cluster



# show_player_and_bag_pos_logs("./Metapunch Data/user_2023_11_15_16_19_43_116360.json","./bag.png")
# show_2D_playerRHandPosLogs("./Metapunch Data/user_2023_11_15_16_19_43_116360.json")
# show_playerRHandPosLogs("./Metapunch Data/user_2023_11_15_16_19_43_116360.json")# 全部右手和沙包數值3全部數值3D畫
# test_show_playerRHandPosLogs("user_2024_9_14_15_7_7__error") #顯示右手打到的沙包密集程度



# process_new_file("./processed_users1/user_2024_10_12_16_21_31__error.json") #顯示8種比較圖

# show_playerRHandLHandPosLogs("./processed_users1/user_2024_10_12_16_21_31__error.json")#2D圖片
# show_persecond_punch_Logs("./processed_users1/user_2024_10_12_16_21_31__error.json")#有處理過楨數的圖
# show_playerRHand_LHandPosLogs("./processed_users1/user_2024_10_12_16_21_31__error.json")#無處理楨數的圖
# show_head_and_bag_pos_logs("./processed_users1/user_2024_8_15_14_53_44__error.json")#顯示頭部與沙包位置
# show_boxplot_punch_reaction_power("./processed_users1/user_2024_10_12_16_20_10_.json")
# animate_playerRHand_LHandPosLogs("./processed_users1/user_2024_10_12_16_21_31__error.json",'all')


# show_guardPosLogs("./processed_users1/user_2024_10_12_16_21_31_.json")
# show_plot_punch_and_reaction("./processed_users1/user_2024_8_15_17_9_6_.json")#顯示三種連續性數據折線圖

# show_boxingstance("./processed_users1/user__2024_12_4 14_29_12_d4.json")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.total_r_offset = []  # 記錄所有右拳穩定度
        self.total_l_offset = []  # 記錄所有左拳穩定度
        self.percentage_series = None 
        self.punchnumber=None
        self.widgets = {}
        self.title('anylias')
        window_width=self.winfo_screenwidth()
        window_height=self.winfo_screenheight()
        self.window_width=1536
        self.window_height=864

        # left = int((window_width - width)/2)       # 計算左上 x 座標
        # top = int((window_height - height)/2)  
        # self.geometry(f"{window_width//2}x{window_height//2}+{left}+{top}" )
        width = int(self.window_width * 0.98)
        height = int(self.window_height * 0.9)
        
        # 計算居中顯示的左上角座標
        left = int((self.window_width - width) / 2)
        top = int(((self.window_height - height) / 5)-15)
        
        # 設置窗口大小和位置
        self.geometry(f"{width}x{height}+{left}+{top}")
        
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner_frame = tk.Frame(self.canvas)# 內部框架
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", self.on_frame_configure)# 綁定事件更新畫布尺寸
        self.photo_images = []  # 儲存Tkinter的PhotoImage，
        
        self.label_frame  = tk.Frame(self.inner_frame,width=width, height=40)  # 在內部框架裡建立一個能存放label的frame
        self.label_frame.pack(fill='x', expand=False)
        self.label_frame.pack_propagate(False) #阻止裡面的改動影響到frame，讓顯示的東西都固定好位置

        # self.label_frame.pack(fill='x', expand=True)
       
        # self.label = ttk.Label(self.label_frame , text="                                 Please first choose the file", font=("Arial", 16, "bold"))
        self.label = ttk.Label(self.label_frame , text="Please first choose the file", font=("Arial", 16, "bold"))
        # self.label.pack(side='left', padx=600)
        self.label.pack(expand=True)
  
  
        self.button_frame = tk.Frame(self.inner_frame, height=60)
        # self.button_frame.pack(fill='x', expand=True)
        self.button_frame.pack(fill='x', expand=False)
        self.button_frame.pack_propagate(False) 
        
        #這邊往下是處理row=0的版面配置
        
        for i in range(5):
            self.button_frame.grid_columnconfigure(i, weight=1)
        
        self.select_button = tk.Button(self.button_frame, text="Choose the json file", font=("Arial", 16, "bold"), command=self.select_file)
        self.select_button.grid(row=0, column=0, padx=10, pady=10)
        
        self.prev_button = tk.Button(self.button_frame, text="Previous User", font=("Arial", 16, "bold"), command=self.previous_user)
        self.prev_button.grid(row=0, column=1, padx=10, pady=10)

        self.next_button = tk.Button(self.button_frame, text="Next User", font=("Arial", 16, "bold"), command=self.next_user)
        self.next_button.grid(row=0, column=2, padx=10, pady=10)

        # self.show_button = tk.Button(self.button_frame, text="Show Summary Data analysis images", font=("Arial", 16, "bold"), command=self.show_plot)
        self.show_button = tk.Button(self.button_frame, text="Show Summary Data", font=("Arial", 16, "bold"), command=self.show_plot)
        self.show_button.grid(row=0, column=3, padx=10, pady=10)

        self.show_formativedata_button = tk.Button(self.button_frame, text="Show Formative Data", font=("Arial", 16, "bold"), command=self.showformativedata)
        self.show_formativedata_button.grid(row=0, column=4, padx=10, pady=10)
        
        self.show_boxingstyle_button = tk.Button(self.button_frame, text="Show Boxing Style", font=("Arial", 16, "bold"), command=self.showboxingstyle)
        self.show_boxingstyle_button.grid(row=0, column=5, padx=10, pady=10)
        # 設置列的比例，使每個按鈕均勻分佈

        
        self.folder_path = "./processed_users1"
        
        # self.folder_path = "./Params"
        
        self.file_list = []
        self.current_file_index = None
        # 綁定滾動事件
        # self.canvas.bind("<MouseWheel>", self.on_scroll)
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))  
        
    

        
        
    def select_file(self):
        initial_dir = self.folder_path
        self.file_path = filedialog.askopenfilename(title="choose json file", filetypes=[("json文件", "*.json")], initialdir=initial_dir)
        if self.file_path:
            filename = os.path.basename(self.file_path)
            display_text = f"Chose {filename}"[:36]  
            self.label.config(text=display_text)
            self.load_file_list()
            self.current_file_index = self.file_list.index(filename)
    
    
    
    def load_file_list(self):# 載入目錄中的所有文件
        
        self.file_list = sorted([f for f in os.listdir(self.folder_path) if f.endswith(".json")])
    
    def previous_user(self): # 切換到上一位使用者
       
        if self.current_file_index is not None and self.current_file_index > 0:
            self.current_file_index -= 1
            self.select_file_by_index()
            if hasattr(self, "control_frame") and self.control_frame is not None and self.control_frame.winfo_exists():
                self.clear_canvas_content()
                self.showformativedata()
            else:
                self.clear_canvas_content()
                self.show_plot()
    
    def next_user(self):# 切換到下一位使用者
        
        if self.current_file_index is not None and self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            self.select_file_by_index()
            if hasattr(self, "control_frame") and self.control_frame is not None and self.control_frame.winfo_exists():
                self.clear_canvas_content()
                self.showformativedata()
            else:
                self.clear_canvas_content()
                self.show_plot()
                
                
    def get_memory_usage(self): #Get memory usage of current process in MB
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 ** 2                
                
    
    def select_file_by_index(self):# 根據 index 選擇文件
        
        if self.file_list:
            self.file_path = os.path.join(self.folder_path, self.file_list[self.current_file_index])
            self.label.config(text=f"Chose {os.path.basename(self.file_path)}")
        #到這
        
    def show_all_position(self,file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)         
            playerPosLogs = data.get("playerPosLogs", [])
            x_coords_player = [point["x"] for point in playerPosLogs]
            y_coords_player = [point["y"] for point in playerPosLogs]
            z_coords_player = [point["z"] for point in playerPosLogs]
            
            bagPosLogs = data.get("bagPosLogs", [])
            x_coords_bag = [point["x"] for point in bagPosLogs]
            y_coords_bag = [point["y"] for point in bagPosLogs]
            z_coords_bag = [point["z"] for point in bagPosLogs]

            playerRHandPosLogs=data.get("playerRHandPosLogs", [])
            playerLHandPosLogs=data.get("playerLHandPosLogs", [])
            
            x_RHandPos_coords=[point["x"]for point in playerRHandPosLogs]
            y_RHandPos_coords=[point["y"]for point in playerRHandPosLogs]
            z_RHandPos_coords=[point["z"]for point in playerRHandPosLogs]
            
            x_LHandPos_coords=[point["x"]for point in playerLHandPosLogs]
            y_LHandPos_coords=[point["y"]for point in playerLHandPosLogs]
            z_LHandPos_coords=[point["z"]for point in playerLHandPosLogs]
            
            puncherIdx = data.get("puncherIdx", [])
            punchTimeCode = data.get("punchTimeCode", [])
            totaltime=math.ceil(punchTimeCode[-1])          
            if totaltime not in [30, 60]:
                if abs(60-totaltime)<abs(30-totaltime):
                    totaltime=60
                else:
                    totaltime=30
            sec_num=len(playerRHandPosLogs)//totaltime
            player_num=len(playerPosLogs)//totaltime
            rhand,lhand,r_xavg,r_yavg,r_zavg,r_number=[],[],[],[],[],[]
            for i in range(len(puncherIdx)) :
                if puncherIdx[i]==0:
                    rhand.append(i) 
                else:
                    lhand.append(i)
            start_rhand=0
            print(f"rhand{rhand}")
            for i in rhand:
                start_idx= max(0,round(punchTimeCode[i]*sec_num) - 5)
                end_idx = min(len(playerRHandPosLogs), round(punchTimeCode[i]*sec_num) + 5)
                temp = playerRHandPosLogs[start_idx:end_idx]
                r_number.append(len(temp))

                if len(temp) > 0:  # 檢查 temp 是否有值
                    x_sum = sum(point["x"] for point in temp)
                    y_sum = sum(point["y"] for point in temp)
                    z_sum = sum(point["z"] for point in temp)
                    
                    r_xavg.append(round(x_sum / len(temp), 3))
                    r_yavg.append(round(y_sum / len(temp), 3))
                    r_zavg.append(round(z_sum / len(temp), 3))
                else:
                    print(f"Warning: Empty segment for Right Hand at i={i}, start_rhand={start_rhand}")
                    r_xavg.append(None)
                    r_yavg.append(None)
                    r_zavg.append(None)
                
            l_xavg,l_yavg,l_zavg,l_number=[],[],[],[]
            for i in lhand:
                start_idx = max(0, round(punchTimeCode[i]*sec_num) - 5)
                end_idx = min(len(playerLHandPosLogs), round(punchTimeCode[i]*sec_num) + 5)
                temp = playerLHandPosLogs[start_idx:end_idx]
                l_number.append(len(temp))
                    

                if len(temp) > 0:  # 檢查 temp 是否有值
                    x_sum = sum(point["x"] for point in temp)
                    y_sum = sum(point["y"] for point in temp)
                    z_sum = sum(point["z"] for point in temp)
                    
                    l_xavg.append(round(x_sum / len(temp), 3))
                    l_yavg.append(round(y_sum / len(temp), 3))
                    l_zavg.append(round(z_sum / len(temp), 3))
                else:
                    print(f"Warning: Empty segment for Left Hand at i={i}, sstart_idx={start_idx}")
                    l_xavg.append(None)
                    l_yavg.append(None)
                    l_zavg.append(None)       
                
            bagPosLogs=data.get("bagPosLogs",[])
            x_bagPos_coords=[point["x"] for point in bagPosLogs]
            y_bagPos_coords=[point["y"] for point in bagPosLogs]
            z_bagPos_coords=[point["z"] for point in bagPosLogs]
            
            player_xavg,player_yavg,player_zavg=[],[],[]
            print(f"len(player){len(playerPosLogs)}")
            start_player=0
            for i in punchTimeCode:
                start_idx = max(0, round(i*player_num) - 5)
                end_idx = min(len(playerPosLogs), round(i*player_num) + 5)
                temp = playerPosLogs[start_idx:end_idx]
                if len(temp) > 0:  # 檢查 temp 是否有值
                    x_sum = sum(point["x"] for point in temp)
                    y_sum = sum(point["y"] for point in temp)
                    z_sum = sum(point["z"] for point in temp)
                    
                    player_xavg.append(round(x_sum / len(temp), 3))
                    player_yavg.append(round(y_sum / len(temp), 3))
                    player_zavg.append(round(z_sum / len(temp), 3))
                else:
                    print(f"Warning: Empty segment for Player at i={i}, start_player={start_player}")
                    player_xavg.append(None)
                    player_yavg.append(None)
                    player_zavg.append(None)        
            
            
            
            fig=plt.figure()
            ax=fig.add_subplot(111,projection='3d')

            ax.scatter(r_xavg,r_zavg,r_yavg,color="darkturquoise",s=30,label="Right hand")
            ax.scatter(l_xavg,l_zavg,l_yavg,color="lime",s=30,label="Left hand")
            ax.scatter(player_xavg,player_zavg,player_yavg,color="violet",s=30,label="Head Position")  
            
               
            ax.scatter(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",s=0.5,label="All Right hand") #全部的右手點
            ax.scatter(x_LHandPos_coords,z_LHandPos_coords,y_LHandPos_coords,color="green",s=0.5,label="All Left hand") #全部的左手點
            center_x= x_coords_bag[0] 
            center_y = z_coords_bag[0] 
            radius=0.05            
            max_num= max(max(y_coords_player),max(y_LHandPos_coords),max(y_RHandPos_coords))
            self.plot_cylinder(ax,center_x,center_y,radius,max_num, color='red')            

                          
            ax.scatter(x_coords_player, z_coords_player, y_coords_player, color='purple', s=0.5, label='All Head Position')
            ax.scatter([x_coords_player[0]], [z_coords_player[0]], [y_coords_player[0]], color='orange', s=50, label='Start Position',zorder=5)
            ax.scatter([x_coords_player[-1]], [z_coords_player[-1]], [y_coords_player[-1]], color='black', s=50, label='End Position',zorder=5)
            ax.scatter(x_coords_bag,z_coords_bag,y_coords_bag,color="red",s=8,label="Bag") 
            # ax.set_box_aspect([1, 1, 1]) 
            # ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='.', color='blue', label=' Head Position')   
            # ax.plot([z_coords_player[0]], [x_coords_player[0]], [y_coords_player[0]], marker='o', color='yellow', markersize=5,label='Start Position')
            # ax.plot([z_coords_player[-1]], [x_coords_player[-1]], [y_coords_player[-1]],marker='o', color='black', markersize=5,label='End Position') 
            # ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='.', color='red', label='Bag Position')  
                      
            ax.set_title("All Position")

            ax.set_xlim(min(min(x_RHandPos_coords),min(x_LHandPos_coords),min(x_coords_bag),min(x_coords_player),center_x-2*radius),max(max(x_RHandPos_coords),max(x_LHandPos_coords),max(x_coords_bag),max(x_coords_player),center_x+2*radius))
            ax.set_ylim(min(min(z_RHandPos_coords),min(z_LHandPos_coords),min(z_coords_bag),min(z_coords_player),center_y-2*radius),max(max(z_RHandPos_coords),max(z_LHandPos_coords),max(z_coords_bag),max(z_coords_player),center_y+2*radius))
            ax.set_zlim(min(min(y_RHandPos_coords),min(y_LHandPos_coords),min(y_coords_bag),min(y_coords_player)),max(max(y_RHandPos_coords),max(y_LHandPos_coords),max(y_coords_bag),max(y_coords_player)))
            
            # ax.set_ylim(0.15,0.2)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Z Axis')                                                                                 
            ax.set_zlabel('Y Axis')
            ax.legend(loc='upper left', bbox_to_anchor=(0.7, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,    fontsize='small')
            ax.view_init(elev=37,azim=-64)#拿來調整最後圖的旋轉
            self.canvas_all_position=FigureCanvasTkAgg(fig, master=self.canvas)
            self.canvas_all_position.draw()
            self.window_all_position=self.canvas.create_window(375, 390, anchor="nw", window=self.canvas_all_position.get_tk_widget(), width=365, height=350)
            
    def show_head_and_bag_pos_logs(self,file_path):#使用者head位置與沙包位置3d
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
            playerPosLogs = data.get("playerPosLogs", [])
            x_coords_player = [point["x"] for point in playerPosLogs]
            y_coords_player = [point["y"] for point in playerPosLogs]
            z_coords_player = [point["z"] for point in playerPosLogs]
            
            bagPosLogs = data.get("bagPosLogs", [])
            x_coords_bag = [point["x"] for point in bagPosLogs]
            y_coords_bag = [point["y"] for point in bagPosLogs]
            z_coords_bag = [point["z"] for point in bagPosLogs]
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='.', color='blue', label=' Head Position')   
            ax.plot([z_coords_player[0]], [x_coords_player[0]], [y_coords_player[0]], marker='o', color='yellow', markersize=5,label='Start Position')
            ax.plot([z_coords_player[-1]], [x_coords_player[-1]], [y_coords_player[-1]],marker='o', color='black', markersize=5,label='End Position') 
                    
            ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='.', color='red', label='Bag Position') 
            # ax.plot(z_coords_player, x_coords_player, y_coords_player, marker='o', markersize=2, color='blue', label='Player Position')   
            # ax.plot(z_coords_bag, x_coords_bag, y_coords_bag, marker='o', markersize=2, color='red', label='Bag Position')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            plt.title("The absolute position of the Head and the Bag")
            ax.set_xlabel('Z Axis')
            ax.set_ylabel('X Axis')
            ax.set_zlabel('Y Axis')
            # ax.set_xlim(19.3,19.8)#這裡設定X值會出錯要再研究
            ax.legend(loc='upper left', bbox_to_anchor=(0.83, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,fontsize='small')
            ax.view_init(elev=62, azim=150)  
            # canvas_absolute_position = FigureCanvasTkAgg(fig, master=self.canvas)
            # canvas_absolute_position.draw()
            # self.window_headbag_position=self.canvas.create_window(921, 90, anchor="nw", window=canvas_absolute_position.get_tk_widget(), width=400, height=300)
            self.canvas_widget_headbag_position = FigureCanvasTkAgg(fig, master=self.canvas)
            self.canvas_widget_headbag_position.draw()
            self.window_headbag_position = self.canvas.create_window(921, 90, anchor="nw", window=self.canvas_widget_headbag_position.get_tk_widget(), width=400, height=300)      
            #1233    
      
    def show_playerRHand_LHandPosLogs(self,file_path):#沒有處理平均值的，會顯示每個frame
        with open(file_path,"r",encoding="utf-8") as file:
            data=json.load(file)
            playerRHandPosLogs=data.get("playerRHandPosLogs", [])
            playerLHandPosLogs=data.get("playerLHandPosLogs", [])
            
            x_RHandPos_coords=[point["x"]for point in playerRHandPosLogs]
            y_RHandPos_coords=[point["y"]for point in playerRHandPosLogs]
            z_RHandPos_coords=[point["z"]for point in playerRHandPosLogs]
            
            x_LHandPos_coords=[point["x"]for point in playerLHandPosLogs]
            y_LHandPos_coords=[point["y"]for point in playerLHandPosLogs]
            z_LHandPos_coords=[point["z"]for point in playerLHandPosLogs]       
            
            bagPosLogs=data.get("bagPosLogs",[])
            x_bagPos_coords=[point["x"] for point in bagPosLogs]
            y_bagPos_coords=[point["y"] for point in bagPosLogs]
            z_bagPos_coords=[point["z"] for point in bagPosLogs]
            fig=plt.figure()
            fig=Figure(figsize=(10,10)) #調整大小
            ax=fig.add_subplot(111,projection='3d')
            # ax.plot(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",marker='o',markersize=3)
            # ax.plot(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",marker='o',markersize=3)
            ax.scatter(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",s=8,label="Right hand")
            ax.scatter(x_LHandPos_coords,z_LHandPos_coords,y_LHandPos_coords,color="green",s=8,label="Left hand")
            ax.scatter(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",s=8,label="Bag")
            ax.set_title("The position of the both hands and the bag")
            # ax.set_xlim(min(x_RHandPos_coords),max(x_LHandPos_coords))
            ax.set_xlim(min(min(x_RHandPos_coords),min(x_LHandPos_coords)),max(max(x_RHandPos_coords),max(x_LHandPos_coords),max(x_bagPos_coords)))
            # ax.set_ylim(0.15,0.2)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Z Axis')                                                                                 
            ax.set_zlabel('Y Axis')
            ax.legend(loc='upper left', bbox_to_anchor=(0.83, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,    fontsize='small')
            ax.view_init(elev=37,azim=-64)#拿來調整最後圖的旋轉
            self.canvas_widget = FigureCanvasTkAgg(fig, master=self.canvas)
            self.canvas_widget.draw()
            self.window_both_position=self.canvas.create_window(890, 390, anchor="nw", window=self.canvas_widget.get_tk_widget(), width=430, height=350)
            #1263
            
    # def show_persecond_punch_Logs(self,file_path):#有處理平均值的，也可稱作每秒出拳視覺圖，讓使用者知道那一秒你的手在哪邊，目前預設30
    #     with open(file_path,"r",encoding="utf-8") as file:
    #         data=json.load(file)
    #         playerRHandPosLogs=data.get("playerRHandPosLogs", [])
    #         playerLHandPosLogs=data.get("playerLHandPosLogs", [])
    #         avg_playerRHandPosLogs=[]
    #         avg_playerLHandPosLogs=[]
            
    #         for i in range(0,len(playerRHandPosLogs),30):
    #             each=playerRHandPosLogs[i:i+30]
    #             avg_x = round(np.mean([point["x"] for point in each]), 4)
    #             avg_y = round(np.mean([point["y"] for point in each]), 4)  
    #             avg_z = round(np.mean([point["z"] for point in each]), 4)
    #             avg_playerRHandPosLogs.append({"x":avg_x,"y":avg_y,"z":avg_z}) 
    #         x_RHandPos_coords=[point["x"]for point in avg_playerRHandPosLogs]
    #         y_RHandPos_coords=[point["y"]for point in avg_playerRHandPosLogs]
    #         z_RHandPos_coords=[point["z"]for point in avg_playerRHandPosLogs]
    #         # print(f"max RHandPos_coords {max(x_RHandPos_coords)}")
    #         # print(f"min RHandPos_coords {min(x_RHandPos_coords)}")
    #         for i in range(0,len(playerLHandPosLogs),30):
    #             each=playerLHandPosLogs[i:i+30]
    #             avg_x = round(np.mean([point["x"] for point in each]), 4)
    #             avg_y = round(np.mean([point["y"] for point in each]), 4)  
    #             avg_z = round(np.mean([point["z"] for point in each]), 4)
    #             avg_playerLHandPosLogs.append({"x":avg_x,"y":avg_y,"z":avg_z}) 
    #         x_LHandPos_coords=[point["x"]for point in avg_playerLHandPosLogs]
    #         y_LHandPos_coords=[point["y"]for point in avg_playerLHandPosLogs]
    #         z_LHandPos_coords=[point["z"]for point in avg_playerLHandPosLogs]       
    #         # print(f"max LHandPos_coords {max(x_LHandPos_coords)}")
    #         # print(f"min LHandPos_coords {min(x_LHandPos_coords)}")
            
    #         bagPosLogs=data.get("bagPosLogs",[])
    #         avg_bagPosLogs=[]
    #         x_bagPos_coords=[point["x"] for point in bagPosLogs]
    #         y_bagPos_coords=[point["y"] for point in bagPosLogs]
    #         z_bagPos_coords=[point["z"] for point in bagPosLogs]
    #         fig=plt.figure()
    #         ax=fig.add_subplot(111,projection='3d')
    #         # ax.plot(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",marker='o',markersize=3)
    #         # ax.plot(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",marker='o',markersize=3)
    #         ax.scatter(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",s=4,label="Right hand")
    #         ax.scatter(x_LHandPos_coords,z_LHandPos_coords,y_LHandPos_coords,color="green",s=4,label="Left hand")
    #         ax.scatter(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",s=4,label="Bag")
    #         ax.set_title("Punch position point per second")
    #         # ax.set_xlim(min(x_LHandPos_coords),max(x_RHandPos_coords))
    #         ax.set_xlim(min(min(x_RHandPos_coords),min(x_LHandPos_coords)),max(max(x_RHandPos_coords),max(x_LHandPos_coords),max(x_bagPos_coords)))
    #         # ax.set_ylim(0.15,0.2)
    #         ax.set_xlabel('X Axis')
    #         ax.set_ylabel('Z Axis')                                                                                 
    #         ax.set_zlabel('Y Axis')
    #         ax.legend(loc='upper left', bbox_to_anchor=(0.83, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,fontsize='small')
    #         ax.view_init(elev=37,azim=-64)#拿來調整最後圖的旋轉
    #         plt.savefig("Punch position point per second.png")
    #         self.canvas_punch_Logs = FigureCanvasTkAgg(fig, master=self.canvas)
    #         self.canvas_punch_Logs.draw()
    #         self.window_punch_Logs=self.canvas.create_window(440, 390, anchor="nw", window=self.canvas_punch_Logs.get_tk_widget(), width=450, height=350)
            #1236
    def plot_cylinder(self,ax, center_x, center_y, radius, height, color='red', resolution=50):
        z=np.linspace(0,height,resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = center_x + radius * np.cos(theta_grid)
        y_grid = center_y + radius * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.5)        
            
            
            
            
            
    def show_punch_position(self,file_path):    #準確的每一拳位置
        if file_path:
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            playerRHandPosLogs = data.get("playerRHandPosLogs", [])
            playerLHandPosLogs = data.get("playerLHandPosLogs", [])
            puncherIdx = data.get("puncherIdx", [])
            playerPosLogs=data.get("playerPosLogs",[])
            punchTimeCode = data.get("punchTimeCode", [])
            totaltime=math.ceil(punchTimeCode[-1])
            if totaltime not in [30, 60]:
                if abs(60-totaltime)<abs(30-totaltime):
                    totaltime=60
                else:
                    totaltime=30
            sec_num=len(playerRHandPosLogs)//totaltime
            player_num=len(playerPosLogs)//totaltime
            rhand,lhand,r_xavg,r_yavg,r_zavg,r_number=[],[],[],[],[],[]
            for i in range(len(puncherIdx)) :
                if puncherIdx[i]==0:
                    rhand.append(i) 
                else:
                    lhand.append(i)
            start_rhand=0
            print(f"rhand{rhand}")
            for i in rhand:
                
                # if round(punchTimeCode[i]*sec_num)+5 < len(playerRHandPosLogs):
                #     temp=playerRHandPosLogs[round(punchTimeCode[i]*sec_num)-5:round(punchTimeCode[i]*sec_num)+5]
                #     start_rhand=round(punchTimeCode[i]*sec_num)
                start_idx = max(0, round(punchTimeCode[i]*sec_num) - 5)
                end_idx = min(len(playerRHandPosLogs), round(punchTimeCode[i]*sec_num) + 5)
                temp = playerRHandPosLogs[start_idx:end_idx]
                r_number.append(len(temp))
                if len(temp) > 0:  # 检查 temp 是否为空
                    x_sum = sum(point["x"] for point in temp)
                    y_sum = sum(point["y"] for point in temp)
                    z_sum = sum(point["z"] for point in temp)
                    
                    r_xavg.append(round(x_sum / len(temp), 3))
                    r_yavg.append(round(y_sum / len(temp), 3))
                    r_zavg.append(round(z_sum / len(temp), 3))
                else:
                    print(f"Warning: Empty segment for Right Hand at i={i}, start_rhand={start_rhand}")
                    r_xavg.append(None)
                    r_yavg.append(None)
                    r_zavg.append(None)
                
            l_xavg,l_yavg,l_zavg,l_number=[],[],[],[]
            start_lhand=0
            for i in lhand:
                # if round(punchTimeCode[i]*sec_num)+5 < len(playerLHandPosLogs):
                #     temp=playerLHandPosLogs[round(punchTimeCode[i]*sec_num)-5:round(punchTimeCode[i]*sec_num)+5]
                start_idx = max(0, round(punchTimeCode[i]*sec_num) - 5)
                end_idx = min(len(playerLHandPosLogs), round(punchTimeCode[i]*sec_num) + 5)
                temp = playerLHandPosLogs[start_idx:end_idx]

                l_number.append(len(temp))
                if len(temp) > 0:  # 检查 temp 是否为空
                    x_sum = sum(point["x"] for point in temp)
                    y_sum = sum(point["y"] for point in temp)
                    z_sum = sum(point["z"] for point in temp)
                    
                    l_xavg.append(round(x_sum / len(temp), 3))
                    l_yavg.append(round(y_sum / len(temp), 3))
                    l_zavg.append(round(z_sum / len(temp), 3))
                else:
                    print(f"Warning: Empty segment for Left Hand at i={i}, start_lhand={start_lhand}")
                    l_xavg.append(None)
                    l_yavg.append(None)
                    l_zavg.append(None)       
                
            bagPosLogs=data.get("bagPosLogs",[])
            x_bagPos_coords=[point["x"] for point in bagPosLogs]
            y_bagPos_coords=[point["y"] for point in bagPosLogs]
            z_bagPos_coords=[point["z"] for point in bagPosLogs]
            
            player_xavg,player_yavg,player_zavg=[],[],[]
            print(f"len(player){len(playerPosLogs)}")


            # for i in range(0,len(punchTimeCode)):
            for i in punchTimeCode:
                # if int(i*player_num)+5<len(playerPosLogs):
                #     temp=playerPosLogs[int(i*player_num)-5:int(i*player_num)+5]
                start_idx = max(0, round(i*player_num) - 5)
                end_idx = min(len(playerPosLogs), round(i*player_num) + 5)
                temp = playerPosLogs[start_idx:end_idx]
                if len(temp) > 0:  # 检查 temp 是否为空
                    x_sum = sum(point["x"] for point in temp)
                    y_sum = sum(point["y"] for point in temp)
                    z_sum = sum(point["z"] for point in temp)
                    
                    player_xavg.append(round(x_sum / len(temp), 3))
                    player_yavg.append(round(y_sum / len(temp), 3))
                    player_zavg.append(round(z_sum / len(temp), 3))
                else:
                    print(f"Warning: Empty segment for Player at i={i}, start_idx={start_idx}")
                    player_xavg.append(None)
                    player_yavg.append(None)
                    player_zavg.append(None)        
            
            
            
            fig=plt.figure()
            ax=fig.add_subplot(111,projection='3d')

            ax.scatter(r_xavg,r_zavg,r_yavg,color="blue",s=20,label="Right hand")
            ax.scatter(l_xavg,l_zavg,l_yavg,color="green",s=20,label="Left hand")
            ax.scatter(player_xavg,player_zavg,player_yavg,color="purple",s=20,label="Head Position")
            
            # def plot_cylinder(self,ax, center_x, center_y, radius, height, color='red', resolution=50):
            #     z=np.linspace(0,height,resolution)
            #     theta = np.linspace(0, 2 * np.pi, resolution)
            #     theta_grid, z_grid = np.meshgrid(theta, z)
            #     x_grid = center_x + radius * np.cos(theta_grid)
            #     y_grid = center_y + radius * np.sin(theta_grid)
            #     ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.5)
                # print(f"x_grid{x_grid}")
                # print(f"y_grid{y_grid}")
              
            center_x= x_bagPos_coords[0] 
            center_y = z_bagPos_coords[0] 
            radius=0.05
            print(f"max(r_yavg){max(r_yavg)}")

            max_r_yavg = max([v for v in r_yavg if v is not None], default=0)
            max_l_yavg = max([v for v in l_yavg if v is not None], default=0)
            max_player_yavg = max([v for v in player_yavg if v is not None], default=0)

            # 计算 max_num
            max_num = max(max_r_yavg, max_l_yavg, max_player_yavg)
            print(f"max(r_yavg): {max_r_yavg}")
            print(f"max(l_yavg): {max_l_yavg}")
            print(f"max(player_yavg): {max_player_yavg}")
            print(f"Calculated max_num: {max_num}")
            # max_num=(max(max(r_yavg),max(l_yavg),max(player_yavg)))
            height=round((max_num),1)
            # print(f"max(player_yavg){max(player_yavg)}")
            print(f"height{height}")
            self.plot_cylinder(ax, center_x, center_y, radius, height, color='red')
            ax.scatter(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",s=20,label="Bag")
            # 
            
            ax.set_title("Punch Position ")
           
            ax.set_xlim(min(min(r_xavg),min(l_xavg),min(x_bagPos_coords),min(player_xavg),(center_x - 2*radius)), max(max(r_xavg),max(l_xavg),max(x_bagPos_coords),max(player_xavg),(center_x + 2*radius)))
            ax.set_ylim(min(min(r_zavg),min(l_zavg),min(player_zavg),center_y- 2*radius),max(max(r_zavg),max(l_zavg),max(player_zavg),center_y+ 2*radius))
            ax.set_zlim( 0, max(max(r_yavg),max(l_yavg),max(player_yavg)))
          
            # x_min = min(min(r_xavg), min(l_xavg), min(x_bagPos_coords), min(player_xavg),(center_x - 2*radius))
            # x_max = max(max(r_xavg), max(l_xavg), max(x_bagPos_coords), max(player_xavg),(center_x + 2*radius))
            # z_min = min(min(r_zavg), min(l_zavg), min(player_zavg),(center_y - 2*radius))
            # z_max = max(max(r_zavg), max(l_zavg), max(player_zavg),(center_y + 2*radius))
            # y_min = 0
            # y_max = max(max(r_yavg), max(l_yavg), max(player_yavg))
            # ax.set_xlim(x_min, x_max)
            # ax.set_ylim(z_min, z_max)
            # ax.set_zlim(y_min, y_max)
            # interval = 0.2
            # x_ticks = np.arange(start=x_min, stop=x_max + interval, step=interval)
            # z_ticks = np.arange(start=z_min, stop=z_max + interval, step=interval)
            # ax.set_xticks(x_ticks)
            # ax.set_zticks(z_ticks)

            ax.set_xlabel('X Axis')
            ax.set_ylabel('Z Axis')                                                                                 
            ax.set_zlabel('Y Axis')
            ax.legend(loc='upper left', bbox_to_anchor=(0.7, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,fontsize='small')
            ax.view_init(elev=37,azim=-64)#拿來調整最後圖的旋轉
            
            ax.set_box_aspect((1, 1, 1))
            # ax.set_aspect('equal')#把坐標軸大小都調整一樣
            
            plt.savefig("Punch position.png")
            self.canvas_punch_Logs = FigureCanvasTkAgg(fig, master=self.canvas)
            self.canvas_punch_Logs.draw()
            self.window_punch_Logs=self.canvas.create_window(1105, 390, anchor="nw", window=self.canvas_punch_Logs.get_tk_widget(), width=365, height=350)

    def show_guard_position(self,file_path):    #準確的每拳防守位置
        if file_path:
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            playerRHandPosLogs = data.get("playerRHandPosLogs", [])
            playerLHandPosLogs = data.get("playerLHandPosLogs", [])
            puncherIdx = data.get("puncherIdx", [])
            punchTimeCode = data.get("punchTimeCode", [])
            reactionTime = data.get("reactionTime", [])
            sec_num=len(playerRHandPosLogs)//30 
            rhand,lhand,r_xavg,r_yavg,r_zavg,r_number=[],[],[],[],[],[]
            for i in range(len(puncherIdx)) :
                if puncherIdx[i]==0:
                    rhand.append(i) 
                else:
                    lhand.append(i)
            for i in rhand:
                temp=playerRHandPosLogs[:round((punchTimeCode[i]+(reactionTime[i]/2))*sec_num)]
                r_number.append(len(temp))
                x_sum=sum(point["x"] for point in temp)
                y_sum=sum(point["y"] for point in temp)
                z_sum=sum(point["z"] for point in temp)
                
                r_xavg.append(round(x_sum/len(temp),3))
                r_yavg.append(round(y_sum/len(temp),3))
                r_zavg.append(round(z_sum/len(temp),3))
                
            l_xavg,l_yavg,l_zavg,l_number=[],[],[],[]
            for i in lhand:
                temp=playerLHandPosLogs[:round((punchTimeCode[i]+(reactionTime[i]/2))*sec_num)]
                # print(f"punchtimenum{punchTimeCode[i]}")
                l_number.append(len(temp))
                x_sum=sum(point["x"] for point in temp)
                y_sum=sum(point["y"] for point in temp)
                z_sum=sum(point["z"] for point in temp)
                
                l_xavg.append(round(x_sum/len(temp),3))
                l_yavg.append(round(y_sum/len(temp),3))
                l_zavg.append(round(z_sum/len(temp),3))
            bagPosLogs=data.get("bagPosLogs",[])
            x_bagPos_coords=[point["x"] for point in bagPosLogs]
            y_bagPos_coords=[point["y"] for point in bagPosLogs]
            z_bagPos_coords=[point["z"] for point in bagPosLogs]
            fig=plt.figure()
            ax=fig.add_subplot(111,projection='3d')
            # ax.plot(x_RHandPos_coords,z_RHandPos_coords,y_RHandPos_coords,color="blue",marker='o',markersize=3)
            # ax.plot(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",marker='o',markersize=3)
            ax.scatter(r_xavg,r_zavg,r_yavg,color="blue",s=10,label="Right hand")
            ax.scatter(l_xavg,l_zavg,l_yavg,color="green",s=10,label="Left hand")
            ax.scatter(x_bagPos_coords,z_bagPos_coords,y_bagPos_coords,color="red",s=4,label="Bag")
            # print(f"Right Hand Points: {len(r_xavg)}")
            # print(f"Left Hand Points: {len(l_xavg)}")
            # print(f"Bag Points: {len(x_bagPos_coords)}")
            ax.set_title("Punch position ")
            # ax.set_xlim(min(x_LHandPos_coords),max(x_RHandPos_coords))
            ax.set_xlim(min(min(r_xavg,),min(l_xavg,),min(x_bagPos_coords)),max(max(r_xavg,),max(l_xavg,),max(x_bagPos_coords)))
            # ax.set_ylim(min(min(r_zavg,),min(l_zavg,)),max(max(r_zavg,),max(l_zavg,),max(x_bagPos_coords)))
            # ax.set_ylim(0.15,0.2)
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Z Axis')                                                                                 
            ax.set_zlabel('Y Axis')
            ax.legend(loc='upper left', bbox_to_anchor=(0.83, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,fontsize='small')
            ax.view_init(elev=37,azim=-64)#拿來調整最後圖的旋轉
            plt.savefig("Punch position.png")
            plt.show()
            # self.canvas_punch_Logs = FigureCanvasTkAgg(fig, master=self.canvas)
            # self.canvas_punch_Logs.draw()
            # self.window_punch_Logs=self.canvas.create_window(200, 390, anchor="nw", window=self.canvas_punch_Logs.get_tk_widget(), width=483, height=350)

           
           
           
    # def show_plot_punch_and_reaction(self,folder_path): #顯示每拳的拳擊速度與反應時間，以折線圖和方格圖的方式進行比較
    def show_plot_punch_and_reaction(self,file_path):
        if self.file_path:
            with open(self.file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
            speed_data=data.get("punchSpeed",[])
            power_data=data.get("punchPower",[])
            reaction_data=data.get("reactionTime",[])
            self.punchnumber=len(reaction_data)
            print(f"reactiondata{reaction_data}")
            time_data=data.get("punchTimeCode",[])
            print(f"time_data{time_data}")
            totaltime =math.ceil(time_data[-1])+5
            # fig,(ax1,ax2,ax3,ax4,ax5,ax6)=plt.subplots(1,6, figsize=(16, 3))
            fig, axs = plt.subplots(1, 6, figsize=(16, 3), width_ratios=[2, 0.5, 2, 0.5, 2, 0.5])
            ax1, ax2, ax3, ax4, ax5, ax6 = axs
            
            # ax1.plot(speed_data,color='skyblue',linewidth=2, markersize=6)
            ax1.scatter(time_data,speed_data,c='skyblue')
            ax1.set_title("Punch Speed ")
            ax1.set_ylabel("Punchspeed(m/s)")
            ax1.set_xlabel("Second")
            # ax1.set_ylim(0,max(speed_data)*1.2)
            ax1.set_ylim(0,round(9.2698*1.2,2))
            ax1.set_xlim(0,totaltime)
        

            sns.boxplot(data=speed_data, ax=ax2, color="skyblue", vert=True, widths=0.3)
            median_speed = round(np.median(speed_data),2)
            y_max = round(9.2698 * 1.2, 2)
            ax2.set_ylim(0, y_max)
            ax2.text(-0.8, median_speed, f"{median_speed}", color="black", ha="center", va="center")   
            # ax2.set_ylim(0,max(speed_data))
            ax2.set_ylim(0,round(9.2698*1.2,2))
            print(f"max speed{max(speed_data)}")
            
            
            # ax3.plot(power_data,color='plum',linewidth=2, markersize=6)
            ax3.scatter(time_data,power_data,c='plum')
            ax3.set_title("Punch Power ")
            ax3.set_ylabel("Punchpower(pound)")
            ax3.set_xlabel("Second")    
            ax3.set_ylim(0,max(power_data)*1.2)
            ax3.set_ylim(0,round(948.5328*1.2,2))
            ax3.set_xlim(0,totaltime)
            print(f"max punchpower{max(power_data)}")
            
            power_data1=power_data.sort()
            sns.boxplot(data=power_data, ax=ax4, color="plum", vert=True, widths=0.3)
            median_power = round(np.median(power_data),2)
            ax4.text(-0.8, median_power, f"{median_power}", color="black", ha="center", va="center")
            ax4.set_ylim(0,round(948.5328*1.2,2))    
        
            
            # ax5.plot(reaction_data,color='orange',linewidth=2, markersize=6)
            ax5.scatter(time_data,reaction_data,c='orange')
            ax5.set_title("Punch Reaction Time ")
            ax5.set_ylabel("PunchReactionTime(sec)")
            ax5.set_xlabel("Second")
            # ax5.set_ylim(0,max(reaction_data)*1.2) 
            ax5.set_ylim(0,round(2)) 
            ax5.set_xlim(0,totaltime)   
            print(f"max reaction{max(reaction_data)}")

            sns.boxplot(data=reaction_data, ax=ax6, color="orange", vert=True, widths=0.3)
            median_reaction = round(np.median(reaction_data),2)
            ax6.text(-0.8, median_reaction, f"{median_reaction}", color="black", ha="center", va="center")   
            # ax6.set_ylim(0,max(reaction_data))     
            ax6.set_ylim(0,round(2))      

            # plt.tight_layout()
            # plt.savefig("show_plot_punch_and_reaction.png")
            # plt.close()
     
            def zoom(event):
                ax = event.inaxes
                if ax is None or event.xdata is None or event.ydata is None:
                    return
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                xdata = event.xdata
                ydata = event.ydata

                scale_factor = 1.2 if event.button == 'up' else 1 / 1.2
                new_xlim = [xdata - (xdata - cur_xlim[0]) / scale_factor,
                            xdata + (cur_xlim[1] - xdata) / scale_factor]
                new_ylim = [max(0, ydata - (ydata - cur_ylim[0]) / scale_factor),
                            ydata + (cur_ylim[1] - ydata) / scale_factor]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # Drag function
            def pan(event):
                ax = event.inaxes
                if ax is None or event.button != 1:
                    return

                if event.name == "button_press_event":
                    pan.start = (event.xdata, event.ydata, ax.get_xlim(), ax.get_ylim())
                    pan.dragging = True
                elif event.name == "motion_notify_event" and hasattr(pan, "start") and pan.start and pan.dragging:
                    x0, y0, xlim, ylim = pan.start
                    if event.xdata is not None and event.ydata is not None:
                        pan.dx = x0 - event.xdata
                        pan.dy = y0 - event.ydata
                elif event.name == "button_release_event" and hasattr(pan, "start") and pan.start:
                    x0, y0, xlim, ylim = pan.start
                    if hasattr(pan, "dx") and hasattr(pan, "dy"):
                        dx = pan.dx
                        dy = pan.dy
                        ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
                        ax.set_ylim([max(0, ylim[0] + dy), ylim[1] + dy])
                        fig.canvas.draw_idle()
                    pan.start = None
                    pan.dragging = False
                    pan.dx = 0
                    pan.dy = 0

            fig.canvas.mpl_connect('scroll_event', zoom)
            fig.canvas.mpl_connect('button_press_event', pan)
            fig.canvas.mpl_connect('motion_notify_event', pan)
            fig.canvas.mpl_connect('button_release_event', pan)

            plt.tight_layout()
            plt.savefig("show_plot_punch_and_reaction.png")
            
            self.canvas_three_type_Logs= FigureCanvasTkAgg(fig,master=self.canvas)
            self.canvas_three_type_Logs.draw()
            self.window_three_type_Logs=self.canvas.create_window(10, 95, anchor="nw", window=self.canvas_three_type_Logs.get_tk_widget(), width=1460, height=300)
            
            
            self.scale_PunchSpeed = Scale(self.canvas, from_=0, to=round(9.2698 * 1.2, 2), resolution=0.1, orient=HORIZONTAL, length=300)
            self.scale_PunchPower = Scale(self.canvas, from_=0, to=round(948.5328 * 1.2, 2), resolution=0.1, orient=HORIZONTAL, length=300)
            self.scale_ReactionTime = Scale(self.canvas, from_=0, to=2, resolution=0.1, orient=HORIZONTAL, length=300)

            self.scale_PunchSpeed.set(0)
            self.scale_PunchSpeed.config(highlightthickness=0,bg="white") 
            self.window_scale_PunchSpeed = self.canvas.create_window(400, 350, anchor="nw", window=self.scale_PunchSpeed,width=100, height=50)  

            self.scale_PunchPower.set(0)
            self.scale_PunchPower.config(highlightthickness=0,bg="white") 
            self.window_scale_PunchPower = self.canvas.create_window(880, 350, anchor="nw", window=self.scale_PunchPower,width=100, height=50)  
            
            self.scale_ReactionTime.set(0)
            self.scale_ReactionTime.config(highlightthickness=0,bg="white") 
            self.window_scale_ReactionTime= self.canvas.create_window(1370, 350, anchor="nw", window=self.scale_ReactionTime,width=100, height=50)  

            
            def set_scale_PunchSpeed(val):
                scale_value = float(val)
                adjusted_scale_value = max(scale_value, 0.1)  
                ax1.set_ylim(0, adjusted_scale_value)  
                ax2.set_ylim(0, adjusted_scale_value)                
                fig.canvas.draw_idle()

            def set_scale_PunchPower(val):
                scale_value = float(val)
                adjusted_scale_value = max(scale_value, 0.1)  
                ax3.set_ylim(0, adjusted_scale_value)  
                ax4.set_ylim(0, adjusted_scale_value)                  
                fig.canvas.draw_idle()
                
            def set_scale_Reactiontime(val):
                scale_value = float(val)
                adjusted_scale_value = max(scale_value, 0.1)   
                ax5.set_ylim(0, adjusted_scale_value)  
                ax6.set_ylim(0, adjusted_scale_value)                  
                fig.canvas.draw_idle()
                
            self.scale_PunchSpeed.config(command=set_scale_PunchSpeed)
            self.scale_PunchPower.config(command=set_scale_PunchPower)
            self.scale_ReactionTime.config(command=set_scale_Reactiontime)
            
            # scale.bind("<B1-Motion>>", lambda event: on_scale(scale.get()))
        

    def show_Boxing_Stance_scatter(self,file_path):
        if self.file_path:
            with open(self.file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
            speed_data=data.get("punchSpeed",[])
            power_data=data.get("punchPower",[])
            reaction_data=data.get("reactionTime",[])
            self.punchnumber=len(reaction_data)
            print(f"reactiondata{reaction_data}")
            time_data=data.get("punchTimeCode",[])
            print(f"time_data{time_data}")
            totaltime =math.ceil(time_data[-1])+5
            # fig,(ax1,ax2,ax3,ax4,ax5,ax6)=plt.subplots(1,6, figsize=(16, 3))
            fig, axs = plt.subplots(1, 2, figsize=(5, 3), width_ratios=[2, 0.5])
            ax1, ax2= axs
            
            # ax1.plot(speed_data,color='skyblue',linewidth=2, markersize=6)
            ax1.scatter(time_data,self.total_r_offset,s=15,c= "darkturquoise")
            ax1.scatter(time_data,self.total_l_offset,s=15,c= "limegreen")
            ax1.set_title("Boxing Defensive Position Stability")
            # ax1.set_ylabel("Punchstability")
            ax1.set_xlabel("Second")
            # ax1.set_ylim(0,max(speed_data)*1.2)
            ax1.set_ylim(0,100)
            ax1.set_xlim(0,totaltime)
        

            sns.boxplot(data=self.total_r_offset, ax=ax2, color="darkturquoise", vert=True, widths=0.6)
            self.total_median_r_offset = round(np.median(self.total_r_offset),2)
            y_max = 100
            ax2.set_ylim(0, y_max)
            ax2.text(-0.8, self.total_median_r_offset, f"{self.total_median_r_offset}", color="black", ha="center", va="center")   

            
        

            # plt.tight_layout()
            # plt.savefig("show_plot_punch_and_reaction.png")
            # plt.close()
     
            def zoom(event):
                ax = event.inaxes
                if ax is None or event.xdata is None or event.ydata is None:
                    return
                cur_xlim = ax.get_xlim()
                cur_ylim = ax.get_ylim()
                xdata = event.xdata
                ydata = event.ydata

                scale_factor = 1.2 if event.button == 'up' else 1 / 1.2
                new_xlim = [xdata - (xdata - cur_xlim[0]) / scale_factor,
                            xdata + (cur_xlim[1] - xdata) / scale_factor]
                new_ylim = [max(0, ydata - (ydata - cur_ylim[0]) / scale_factor),
                            ydata + (cur_ylim[1] - ydata) / scale_factor]

                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # Drag function
            def pan(event):
                ax = event.inaxes
                if ax is None or event.button != 1:
                    return

                if event.name == "button_press_event":
                    pan.start = (event.xdata, event.ydata, ax.get_xlim(), ax.get_ylim())
                    pan.dragging = True
                elif event.name == "motion_notify_event" and hasattr(pan, "start") and pan.start and pan.dragging:
                    x0, y0, xlim, ylim = pan.start
                    if event.xdata is not None and event.ydata is not None:
                        pan.dx = x0 - event.xdata
                        pan.dy = y0 - event.ydata
                elif event.name == "button_release_event" and hasattr(pan, "start") and pan.start:
                    x0, y0, xlim, ylim = pan.start
                    if hasattr(pan, "dx") and hasattr(pan, "dy"):
                        dx = pan.dx
                        dy = pan.dy
                        ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
                        ax.set_ylim([max(0, ylim[0] + dy), ylim[1] + dy])
                        fig.canvas.draw_idle()
                    pan.start = None
                    pan.dragging = False
                    pan.dx = 0
                    pan.dy = 0

            fig.canvas.mpl_connect('scroll_event', zoom)
            fig.canvas.mpl_connect('button_press_event', pan)
            fig.canvas.mpl_connect('motion_notify_event', pan)
            fig.canvas.mpl_connect('button_release_event', pan)

            plt.tight_layout()
            plt.savefig("show_plot_punch_and_reaction.png")
            
            self.canvas_BoxingStance_Logs= FigureCanvasTkAgg(fig,master=self.canvas)#放整張照片的
            self.canvas_BoxingStance_Logs.draw()
            self.window_BoxingStance_Logs=self.canvas.create_window(1105, 390, anchor="nw", window=self.canvas_BoxingStance_Logs.get_tk_widget(), width=380, height=350)
            
            
            self.scale_BoxingStance = Scale(self.canvas, from_=0, to=100, resolution=0.1, orient=HORIZONTAL, length=300)

            
            self.scale_BoxingStance.set(0)
            self.scale_BoxingStance.config(highlightthickness=0,bg="white") 
            self.window_scale_BoxingStance= self.canvas.create_window(1400, 685, anchor="nw", window=self.scale_BoxingStance,width=80, height=50)  

            
            def set_scale_scale_BoxingStance(val):
                scale_value = float(val)*1.2
                adjusted_scale_value = max(scale_value, 0.1)  
                ax1.set_ylim(0, adjusted_scale_value)  
                ax2.set_ylim(0, adjusted_scale_value)                
                fig.canvas.draw_idle()

            self.scale_BoxingStance.config(command=set_scale_scale_BoxingStance)


   
        
    def add_widget(self, name, widget):
        """添加小部件到追踪字典，如果已存在，先銷毀舊的小部件。"""
        if isinstance(self.widgets[name], FigureCanvasTkAgg):
            self.widgets[name].get_tk_widget().destroy()
        else:
            self.widgets[name].destroy()
        self.widgets[name] = widget
        
        
    def clear_canvas_content(self):
        """Clear all content on the canvas and reset stored variables."""
        if hasattr(self,"test_label") and self.test_label:
            self.test_label.destroy()
            self.test_label=None
            print("Cleared the gpt  message")
        
        if hasattr(self,"guardposition_frame") and self.guardposition_frame:
            self.guardposition_frame.destroy()
            self.guardposition_frame=None
            print("Cleared guardposition frame=")
        
        if hasattr(self,"boxstance_frame") and self.boxstance_frame:
            self.boxstance_frame.destroy()
            self.boxstance_frame = None
            print("Cleared boxstance frame")
        
        if hasattr(self, "control_frame") and self.control_frame:
            self.control_frame.destroy()
            self.control_frame = None
            print("Cleared control frame")

        if hasattr(self, "photo_images"):
            for photo in self.photo_images:
                self.canvas.delete(photo)
            self.photo_images.clear()
            print("Cleared photo images")

        if hasattr(self, "window_animation") and self.window_animation:
            self.canvas.delete(self.window_animation)
            self.window_animation = None
            print("Cleared window animation")

        if hasattr(self, 'ani') and self.ani:
            self.ani.event_source.stop()  # Stop the animation event source
            del self.ani  # Delete the animation to free memory
            print("Cleared animation")

        if hasattr(self, "canvas_animation") and self.canvas_animation:
            try:
                self.canvas_animation.get_tk_widget().destroy()
                self.canvas_animation = None
                print("Cleared canvas animation")
            except Exception as e:
                print(f"Error clearing canvas animation: {e}")

        # if hasattr(self, "window_headbag_position") and self.window_headbag_position:
        #     self.canvas.delete(self.window_headbag_position)
        #     self.window_headbag_position = None
        #     print("Cleared headbag position window")

        # if hasattr(self, "canvas_widget_headbag_position") and self.canvas_widget_headbag_position:
        #     try:
        #         self.canvas_widget_headbag_position.get_tk_widget().destroy()
        #         self.canvas_widget_headbag_position = None
        #         print("Cleared headbag position canvas")
        #     except Exception as e:
        #         print(f"Error clearing headbag position canvas: {e}")

        if hasattr(self, "window_punch_Logs") and self.window_punch_Logs:
            self.canvas.delete(self.window_punch_Logs)
            self.window_punch_Logs = None
            print("Cleared punch logs window")

        if hasattr(self, "canvas_punch_Logs") and self.canvas_punch_Logs:
            try:
                self.canvas_punch_Logs.get_tk_widget().destroy()
                self.canvas_punch_Logs = None
                print("Cleared punch logs canvas")
            except Exception as e:
                print(f"Error clearing punch logs canvas: {e}")


        if hasattr(self, "window_all_position") and self.window_all_position:
            self.canvas.delete(self.window_all_position)
            self.window_all_position = None
            print("Cleared all position window")

        if hasattr(self, "canvas_all_position") and self.canvas_all_position:
            self.canvas_all_position.get_tk_widget().destroy()
            self.canvas_all_position = None
            print("Cleared all position canvas")


        if hasattr(self,"window_BoxingStance_Logs") and self.window_BoxingStance_Logs:
            self.canvas.delete(self.window_BoxingStance_Logs)
            self.window_BoxingStance_Logs = None
            print("Cleared all self.window_BoxingStance_Logs")

        if hasattr(self,"canvas_BoxingStance_Logs") and self.canvas_BoxingStance_Logs:
            self.canvas_BoxingStance_Logs.get_tk_widget().destroy()
            self.canvas_BoxingStance_Logs=None
            print("Cleared all self.canvas_BoxingStance_Logs")
            
        if hasattr(self,"window_scale_BoxingStance") and self.window_scale_BoxingStance:
            self.canvas.delete(self.window_scale_BoxingStance)
            self.window_scale_BoxingStance = None
            print("Cleared all self.window_scale_BoxingStance")

        if hasattr(self,"scale_BoxingStance") and self.scale_BoxingStance:
            self.scale_BoxingStance.destroy()
            self.scale_BoxingStance=None
            print("Cleared all self.scale_BoxingStance")



            #這邊世上三張圖要刪的，有使用到window的命名方式都需要使用self.canvs 去刪除掉
        if hasattr(self,"window_three_type_Logs") and self.window_three_type_Logs:
            self.canvas.delete(self.window_three_type_Logs)
            self.window_three_type_Logs = None
            print("Cleared all three types punchpower logs window")
        
        if hasattr(self,"canvas_three_type_Logs") and self.canvas_three_type_Logs:
            self.canvas_three_type_Logs.get_tk_widget().destroy()
            self.canvas_three_type_Logs=None
            print("Cleared all three types punchpower logs canvas")
            
            
        if hasattr(self, "window_scale_PunchSpeed") and self.window_scale_PunchSpeed:
            self.canvas.delete(self.window_scale_PunchSpeed)
            self.window_scale_PunchSpeed = None
            print("Cleared PunchSpeed Scale window")  
        if hasattr(self,"scale_PunchSpeed") and self.scale_PunchSpeed:  
            self.scale_PunchSpeed.destroy()
            self.scale_PunchSpeed=None
            print("Cleared PunchSpeed Scale ")
    

        if hasattr(self, "window_scale_PunchPower") and self.window_scale_PunchPower:
            self.canvas.delete(self.window_scale_PunchPower)
            self.window_scale_PunchPower = None
            print("Cleared PunchPower Scale window")
        if hasattr(self,"scale_PunchPower") and self.scale_PunchPower:  
            self.scale_PunchPower.destroy()
            self.scale_PunchPower=None
            print("Cleared PunchPower Scale ")


        if hasattr(self, "window_scale_ReactionTime") and self.window_scale_ReactionTime:
            self.canvas.delete(self.window_scale_ReactionTime)
            self.window_scale_ReactionTime = None
            print("Cleared ReactionTime Scale window") 
        if hasattr(self,"scale_ReactionTime") and self.scale_ReactionTime:  
            self.scale_ReactionTime.destroy()
            self.scale_ReactionTime=None
            print("Cleared ReactionTime Scale ")        
            
            
        # if hasattr(self,"window_boxstance") and self.window_boxstance:
        #     self.canvas.delete(self.window_boxstance)
        #     self.window_boxstance = None
        #     print("Cleared window boxstance")
            
        # if hasattr(self, "canvas_boxstance") and self.canvas_boxstance:
        #     self.canvas_boxstance.get_tk_widget().destroy()
        #     self.canvas_boxstance=None
        #     print("Cleared canvas boxstance") 
            
            
        if hasattr(self,"window_newboxstance") and self.window_newboxstance:
            self.canvas.delete(self.window_newboxstance) 
            self.window_newboxstance = None
            print("Cleared window newboxstance") 
            
        if hasattr(self, "canvas_newboxstance") and self.canvas_newboxstance:   
            self.canvas_newboxstance.get_tk_widget().destroy()
            self.canvas_newboxstance=None
            print("Cleared canvas  newboxstance")
            
        # Close any remaining Matplotlib figures
        plt.close('all')
        print("Closed all Matplotlib figures window")

        # Update canvas scroll region
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height()))
            

 
        #下面是原本圖片
        # if hasattr(self, "window_both_position") and self.window_both_position:
        #     self.canvas.delete(self.window_both_position)
        #     self.window_both_position = None  
        # if hasattr(self, "canvas_widget") and self.canvas_widget: #The position of the both hands and the bag
        #     try:
        #         self.canvas_widget.get_tk_widget().destroy()
        #         self.canvas_widget = None 
        #     except Exception as e:
        #         print(f"Error in cThe position of the both hands and the bag: {e}")
            

        # self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height())) 
            
            
    
    def show_plot(self):
        if self.file_path:
            try:
                self.clear_canvas_content()
                initial_memory = self.get_memory_usage()
                print(f"Initial memory usage: {initial_memory:.2f} MB")
                process_new_file(self.file_path)  
                new_row = boxing_df.iloc[-1]  
                # plot_data(new_row) 
                self.percentage_series = plot_data(new_row)
                print("Percentage Series:", self.percentage_series)
                
                # show_playerRHandLHandPosLogs(self.file_path) 

                img1= Image.open("seven_summing.png")
                # img2= Image.open("Analysis of the different hands of punches in the hit regions.png")

                img1 = img1.resize((1100, 900), Image.LANCZOS)  
                # img2 = img2.resize((350, 300), Image.LANCZOS)  

                photo1= ImageTk.PhotoImage(img1)
                # photo2= ImageTk.PhotoImage(img2)

                self.canvas.create_image(10, 90, anchor="nw", image=photo1)#但表圖會從x=50,y=50的左上角開始顯現出來
                # self.canvas.create_image(1110, 90, anchor="nw", image=photo2)
                # self.canvas.create_image(910, 1050, anchor="nw", image=photo2)
                # 保存引用，防止圖片被垃圾回收
                self.photo_images = [ photo1]
                
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height + 900))  # 設置 scrollregion 比實際畫布的範圍大，這樣滾動條會啟用
                final_memory = self.get_memory_usage()
                print(f"Final memory usage: {final_memory:.2f} MB")
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                
                
    def initialize_control_frame(self):
        if hasattr(self, "control_frame") and self.control_frame is not None:
            # 如果 control_frame 已存在，直接返回
            return
        # built the control_frame 
        self.control_frame = tk.Frame(self.canvas)
        
        self.frame_limit_label = tk.Label(self.control_frame, text="Frame Number:", font=("Arial", 16, "bold"))
        self.frame_limit_label.grid(row=0, column=0, padx=1, pady=1)

        self.frame_limit_spinbox_var = tk.StringVar(value="1")
        self.frame_limit_spinbox = tk.Spinbox(
            self.control_frame, from_=1, to=5, textvariable=self.frame_limit_spinbox_var, font=("Arial", 16, "bold"), width=5
        )
        self.frame_limit_spinbox.grid(row=0, column=1, padx=1, pady=1)

        self.frame_all_button = tk.Button(self.control_frame,text="All",font=("Arial", 16, "bold"), command= self.all_button_active)
        self.frame_all_button.grid(row=0, column=2, padx=1, pady=1)


        self.frame_speed_label = tk.Label(self.control_frame, text="Speed Level", font=("Arial", 16, "bold"))
        self.frame_speed_label.grid(row=0, column=3, padx=1, pady=1)

        self.frame_speed_spinbox_var = tk.StringVar(value="1")  # 初始值
        self.frame_speed_spinbox = tk.Spinbox(
            self.control_frame, from_=1, to=5, textvariable=self.frame_speed_spinbox_var, font=("Arial", 16, "bold"), width=5, 
        )
        self.frame_speed_spinbox.grid(row=0, column=4, padx=1, pady=1)  

    def all_button_active(self):
        self.frame_limit_spinbox_var.set("all")

                                    
    def show_animation(self):    
        if self.file_path:
            with open(self.file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                playerRHandPosLogs = data.get("playerRHandPosLogs", [])
                playerLHandPosLogs = data.get("playerLHandPosLogs", [])
                print(len(playerRHandPosLogs))
                print(len(playerLHandPosLogs))

                bagPosLogs = data.get("bagPosLogs", [])
                playerPosLogs = data.get("playerPosLogs",[])
                frame_limit_value=10
                
                # 抽取座標數據
                x_RHandPos_coords = [point["x"] for point in playerRHandPosLogs]
                y_RHandPos_coords = [point["y"] for point in playerRHandPosLogs]
                z_RHandPos_coords = [point["z"] for point in playerRHandPosLogs]

                x_LHandPos_coords = [point["x"] for point in playerLHandPosLogs]
                y_LHandPos_coords = [point["y"] for point in playerLHandPosLogs]
                z_LHandPos_coords = [point["z"] for point in playerLHandPosLogs]

                x_bagPos_coords = [point["x"] for point in bagPosLogs]
                y_bagPos_coords = [point["y"] for point in bagPosLogs]
                z_bagPos_coords = [point["z"] for point in bagPosLogs]
                
                x_playerPos_coords = [point["x"] for point in playerPosLogs]
                y_playerPos_coords = [point["y"] for point in playerPosLogs]
                z_playerPos_coords = [point["z"] for point in playerPosLogs]

                # 創建圖表
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                scatter_r = ax.scatter([], [], [], color="blue", s=20, label="Right hand")
                scatter_l = ax.scatter([], [], [], color="green", s=20, label="Left hand")
                scatter_head = ax.scatter([],[],[],color="purple", s=20, label="Head position")
                line_r, = ax.plot([],[],[],linestyle="--",color="blue",alpha=0.5)#由於他回傳的不是3D物件，必須加上逗號來代表指向他 
                line_l, = ax.plot([], [], [], linestyle="--", color="green", alpha=0.5)
                line_head, = ax.plot([],[],[],linestyle="--", color="purple", alpha=0.5)
                scatter_bag = ax.scatter(x_bagPos_coords, z_bagPos_coords, y_bagPos_coords, color="red", s=20, label="Bag")
                center_x= x_bagPos_coords[0] 
                center_y = z_bagPos_coords[0] 
                radius=0.05            
                max_num= max(max(y_playerPos_coords),max(y_LHandPos_coords),max(y_RHandPos_coords))
                self.plot_cylinder(ax,center_x,center_y,radius,max_num, color='red')

                ax.set_xlim(min(min(x_LHandPos_coords),min(x_playerPos_coords),min(x_RHandPos_coords),min(x_bagPos_coords),center_x-2*radius), max(max(x_LHandPos_coords),max(x_playerPos_coords),max(x_RHandPos_coords),max(x_bagPos_coords),center_x+2*radius))
                ax.set_ylim(min(min(z_LHandPos_coords),min(z_playerPos_coords),min(z_RHandPos_coords),min(z_bagPos_coords),center_y-2*radius), max(max(z_LHandPos_coords),max(z_playerPos_coords),max(z_RHandPos_coords),max(z_bagPos_coords),center_y+2*radius ))
                # ax.set_ylim(min(z_RHandPos_coords + z_LHandPos_coords), max(z_RHandPos_coords + z_LHandPos_coords))
                # ax.set_zlim(min(min(y_LHandPos_coords),min(y_playerPos_coords),min(y_bagPos_coords)), 2)
                ax.set_zlim(min(min(y_LHandPos_coords),min(y_playerPos_coords),min(y_bagPos_coords)), max(max(y_RHandPos_coords),max(y_playerPos_coords),max(y_bagPos_coords)))

                ax.set_title("Punch Animation")
                ax.set_xlabel('X Axis')
                ax.set_ylabel('Z Axis')
                ax.set_zlabel('Y Axis')
                ax.view_init(elev=37, azim=-64)

                ax.legend(loc='upper left', bbox_to_anchor=(0.7, 1), borderaxespad=0,borderpad=0.3,labelspacing=0.2,fontsize='small')
                    
                frame_limit_value=1
                speed_value=150
                # 更新動畫函數
                frame_delays = []
                last_frame_time = [time.time()] 
                def update(frame,frame_limit_value):
                    current_time=time.time()
                    delay = (current_time - last_frame_time[0]) * 1000
                    frame_delays.append(delay)
                    last_frame_time[0] = current_time
                    
                    # if len(frame_delays)==200:#每累積200個delay數量，顯示一次目前每個frame的delay
                    #     print("Frame Delays (200 values):")
                    #     print(frame_delays)
                        
                    #     self.ani.event_source.stop()  
                    #     del self.ani 
                    # frame_delays=[0]
                    # frame_limit = len(playerRHandPosLogs) if frame_limit_value == "all" else int(frame_limit_value)
                    if frame_limit_value == "all":
                        current_index = frame + 1
                        start_index = 0
                    else:
                        frame_limit=int(frame_limit_value)*5 #一次產生5個點
                        current_index = min(len(x_RHandPos_coords), frame + frame_limit)
                        start_index = max(0, current_index - frame_limit)
                    # print(f"Frame {frame}, Start Index: {start_index}, Current Index: {current_index}, Frame Limit: {frame_limit}")
                    # print(f"Total Points in Right Hand Dataset: {len(x_RHandPos_coords)}")
                    # print(f"Line Points (Right Hand): {len(x_RHandPos_coords[start_index:current_index])}")
                    # print(f"Total Points in Left Hand Dataset: {len(x_LHandPos_coords)}")
                    # print(f"Line Points (Right Hand): {len(x_LHandPos_coords[start_index:current_index])}")

                    line_r.set_data(x_RHandPos_coords[start_index:current_index-1],z_RHandPos_coords[start_index:current_index-1])    
                    line_r.set_3d_properties(y_RHandPos_coords[start_index:current_index-1])

                    line_l.set_data(x_LHandPos_coords[start_index:current_index-1], z_LHandPos_coords[start_index:current_index-1])
                    line_l.set_3d_properties(y_LHandPos_coords[start_index:current_index-1])

                    line_head.set_data(x_playerPos_coords[start_index:current_index-1],z_playerPos_coords[start_index:current_index-1])
                    line_head.set_3d_properties(y_playerPos_coords[start_index:current_index-1])
                    
                    scatter_r._offsets3d = ([x_RHandPos_coords[current_index-1]],
                                            [z_RHandPos_coords[current_index-1]],
                                            [y_RHandPos_coords[current_index-1]])
                    
                    
                    scatter_l._offsets3d = ([x_LHandPos_coords[current_index-1]],
                                            [z_LHandPos_coords[current_index-1]],
                                            [y_LHandPos_coords[current_index-1]])
                    
                    
                    scatter_head._offsets3d = ([x_playerPos_coords[current_index-1]],
                                                [z_playerPos_coords[current_index-1]],
                                                [y_playerPos_coords[current_index-1]])
                    
                    # print(f"Right Hand Line: {len(x_RHandPos_coords[start_index:current_index])}")
                    # print(f"Left Hand Line: {len(x_LHandPos_coords[start_index:current_index])}")
                    # print(f"Right Hand Scatter: {len([x_RHandPos_coords[current_index-1]])}")
                    # print(f"Left Hand Scatter: {len([x_LHandPos_coords[current_index-1]])}")

                    return line_r, line_l, scatter_r, scatter_l, line_head, scatter_head

                # 動畫
                # ani = FuncAnimation(fig, update(frame_limit_value), frames=np.arange(0, len(x_RHandPos_coords)), interval=100, blit=False)
                self.ani = FuncAnimation(fig, partial(update,frame_limit_value=frame_limit_value), frames=np.arange(0, len(x_RHandPos_coords)), interval=200, blit=False)#改了ani
                self.initialize_control_frame() 


                def create_animation(frame_limit_value, speed_value):
                    # 停止並刪除舊的動畫
                    if hasattr(self, 'ani') and self.ani:
                        self.ani.event_source.stop()  
                        del self.ani 
                    frame_delays.clear()
                    # self.ani = FuncAnimation(fig, partial(update, frame_limit_value=frame_limit_value), frames=np.arange(0, len(x_RHandPos_coords)), interval=(round(0.85 ** int(speed_value)*100)), blit=False)
                    # self.ani = FuncAnimation(fig, partial(update, frame_limit_value=frame_limit_value), frames=np.arange(0, len(x_RHandPos_coords)), interval=(200-((int(speed_value)-1)*40)), blit=False)  
                    self.ani = FuncAnimation(fig, partial(update, frame_limit_value=frame_limit_value), frames=np.arange(0, len(x_RHandPos_coords)), interval=(20-((int(speed_value)-1)*2)), blit=False)    
                    print(f"interval={(200-((int(speed_value)-1)*40))}")
                    self.canvas_animation.draw()   
                    

                    # if hasattr(self, 'control_frame') and self.control_frame is not None:
                    #     self.control_frame.destroy()
                    #     self.control_frame = None 

                def on_spinbox_change(*args):
                    # frame_limit_value=10
                    # speed_value=100
                    frame_limit_value=1
                    speed_value=1
                    spinbox_frame_value = (self.frame_limit_spinbox_var.get())
                    print(f"spinbox_frame_value={self.frame_limit_spinbox_var.get()}")
                    spinbox_speed_value = (self.frame_speed_spinbox_var.get())
                    # spinbox_speed_value  = (round(0.85 ** int(self.frame_speed_spinbox_var.get())*100))
                    print(f"spinbox_speed_value={self.frame_speed_spinbox_var.get()}")
                    if spinbox_frame_value.isdigit() or spinbox_frame_value == "all" or spinbox_speed_value.isdigit():
                        if spinbox_frame_value != frame_limit_value:  # 檢查值是否更改
                            frame_limit_value = spinbox_frame_value
                            create_animation(frame_limit_value,speed_value) 
                        print(f"the value{frame_limit_value}")
                        if spinbox_speed_value !=speed_value:
                            speed_value=spinbox_speed_value
                            create_animation(frame_limit_value,speed_value) 
                self.frame_limit_spinbox_var.trace_add("write", on_spinbox_change)
                self.frame_speed_spinbox_var.trace_add("write", on_spinbox_change)
                self.canvas.create_window(10, 700, anchor="nw", window=self.control_frame)

                # 使用 FigureCanvasTkAgg 將圖表嵌入到 Tkinter
                self.canvas_animation = FigureCanvasTkAgg(fig, master=self.canvas)
                self.canvas_animation.draw()
                # self.window_animation=self.canvas.create_window(10, 390, anchor="nw", window=self.canvas_animation.get_tk_widget(), width=483, height=350)

                self.control_frame.place(x=10, y=742)
                self.window_animation=self.canvas.create_window(10, 390, anchor="nw", window=self.canvas_animation.get_tk_widget(), width=365, height=350)
                self.canvas.tag_raise(self.window_animation)

    def initialize_guardposition_frame(self):
        if hasattr(self, "guardposition_frame") and self.guardposition_frame is not None:
            return

        self.guardposition_frame = tk.Frame(self.canvas,width=400, height=10)
        self.guardposition_frame.pack_propagate(False) 
        
        self.frame_righthand_label = tk.Label(self.guardposition_frame, text="Move XYZ", font=("Arial", 12, "bold"))
        self.frame_righthand_label.grid(row=0, column=0, padx=1, pady=1)
        self.frame_righthand_Stability_label = tk.Label(self.guardposition_frame, text="Stability", font=("Arial", 12, "bold"))
        self.frame_righthand_Stability_label.grid(row=0, column=1, padx=1, pady=1)

        self.frame_lefthand_label = tk.Label(self.guardposition_frame, text="Move XYZ", font=("Arial", 12, "bold"))
        self.frame_lefthand_label.grid(row=0, column=2, padx=1, pady=1)
        self.frame_lefthand_Stability_label = tk.Label(self.guardposition_frame, text="Stability", font=("Arial", 12, "bold"))
        self.frame_lefthand_Stability_label.grid(row=0, column=3, padx=1, pady=1)

        self.frame_punchnum_label = tk.Label(self.guardposition_frame, text="Punch Number", font=("Arial", 12, "bold"))
        self.frame_punchnum_label.grid(row=0, column=4, padx=1, pady=1)
        
        self.punchnumber_spinbox_var = tk.StringVar(value="1")
        self.punchnumber_spinbox = tk.Spinbox(
            self.guardposition_frame,
            from_=1,
            to= self.punchnumber,
            textvariable=self.punchnumber_spinbox_var,
            font=("Arial", 12, "bold"),
            width=2
        )
        self.punchnumber_spinbox.grid(row=0, column=5, padx=1, pady=1)

        self.punchnumber_spinbox_var.trace_add("write", self.on_spinbox_change)
        self.canvas.create_window(520, 742, anchor="nw", window=self.guardposition_frame)

    def show_boxingstance(self, file_path):
        if not file_path:
            return

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 資料處理
        playerRHandPosLogs = data.get("playerRHandPosLogs", [])
        playerLHandPosLogs = data.get("playerLHandPosLogs", [])
        playerPosLogs = data.get("playerPosLogs", [])
        puncherIdx = data.get("puncherIdx", [])
        punchTimeCode = data.get("punchTimeCode", [])
        totaltime = math.ceil(punchTimeCode[-1])
        sec_num = len(playerRHandPosLogs) // totaltime #這邊是告訴我每一秒總共會跑幾個FRAME
        print(f"sec_num{sec_num}")

        def sum_array(array, initial_position):
            print(f"Initial position: {initial_position}, Array length: {len(array)}")            
            temp = array[:initial_position]
            print(f"Temp length: {len(temp)}")
            # if len(temp) == 0:
            #     print("Temp is empty. Returning default values.")
            #     return [0], [0], [0]
            x_sum = sum(point["x"] for point in temp)
            y_sum = sum(point["y"] for point in temp)
            z_sum = sum(point["z"] for point in temp)
            return [round(x_sum / len(temp), 3)], [round(y_sum / len(temp), 3)], [round(z_sum / len(temp), 3)]

        initial_position = round(sec_num * 5) #這邊的5是前5秒的意思
        self.r_xavg, self.r_yavg, self.r_zavg = sum_array(playerRHandPosLogs, initial_position)
        self.l_xavg, self.l_yavg, self.l_zavg = sum_array(playerLHandPosLogs, initial_position)
        self.player_xavg, self.player_yavg, self.player_zavg = sum_array(playerPosLogs, initial_position)  


        self.total_r_offset = []  # 記錄所有右拳穩定度
        self.total_l_offset = []  # 記錄所有左拳穩定度
        # 將所有拳擊數據儲存到 self.punch_data
        self.punch_data = [] 
        for idx, j in enumerate(puncherIdx):
            punch_info = {'index': idx, 'hand': 'right' if j == 0 else 'left'}
            start_time = punchTimeCode[idx]
            end_time = totaltime if idx + 1 >= len(punchTimeCode) else punchTimeCode[idx + 1]

            def calculate_offset(logs, start_time, end_time, sec_num, avg_x, avg_y, avg_z):  
                start_idx = round(start_time * sec_num)
                end_idx = round(end_time * sec_num)
                interval_points = logs[start_idx:end_idx]
                if len(interval_points) == 0:  # 確保 interval_points 不為0
                    return 0, 0, 0, 0, 0  # 返回值
                x_avg = sum(point["x"] for point in interval_points) / len(interval_points)#畫面中會移動的x座標
                y_avg = sum(point["y"] for point in interval_points) / len(interval_points)
                z_avg = sum(point["z"] for point in interval_points) / len(interval_points)

                epsilon=1 #代表我把他們使用正規化壓縮在[0,1]，邊長為1的正方形三維空間內
                initial_max_avg_x=avg_x+epsilon#最大值
                initial_max_avg_y=avg_y+epsilon
                initial_max_avg_z=avg_z+epsilon
                
                initial_min_avg_x=avg_x-epsilon#最小值
                initial_min_avg_y=avg_y-epsilon
                initial_min_avg_z=avg_z-epsilon
                
                offset_x = round(x_avg - avg_x, 2)#這邊只是告訴我們她平移動了多少，不參與計算，取道小數點第二位
                offset_y = round(y_avg - avg_y, 2)
                offset_z = round(z_avg - avg_z, 2)
                
                x_normalized = (x_avg - initial_min_avg_x) / (initial_max_avg_x - initial_min_avg_x)#這邊是我在處理normalization的方式，為了算出stability
                y_normalized = (y_avg - initial_min_avg_y) / (initial_max_avg_y - initial_min_avg_y)
                z_normalized = (z_avg - initial_min_avg_z) / (initial_max_avg_z - initial_min_avg_z)

                # 計算歐式距離
                offset_distance = ((x_normalized - 0.5) ** 2 + (y_normalized - 0.5) ** 2 + (z_normalized - 0.5) ** 2) ** 0.5 #0.5為正規化後的中心點

                # 最大距離（對角線長度）
                d_max = 3 ** 0.5

                # 計算穩定度
                offset_rate = round(max(0, 100 * (1 - offset_distance  / d_max)),2 ) # 保證穩定度不小於 0%，1是用來與前面那個100%作用的，因為假設offset_distance=0，代表沒移動，那他穩定度就是100%沒錯
                # offset_distance = round((offset_x**2 + offset_y**2 + offset_z**2) ** 0.5, 3)
                # base_distance = (avg_x**2 + avg_y**2 + avg_z**2) ** 0.5
                # offset_rate = 100 - (round((offset_distance / base_distance) * 100, 2)) if base_distance != 0 else 0
                return offset_x, offset_y, offset_z, offset_distance, offset_rate

            if punch_info['hand'] == 'right':
                r_offset = calculate_offset(
                    playerRHandPosLogs, start_time, end_time, sec_num, self.r_xavg[0], self.r_yavg[0], self.r_zavg[0])
                rl_offset = calculate_offset(
                    playerLHandPosLogs, start_time, end_time, sec_num, self.l_xavg[0], self.l_yavg[0], self.l_zavg[0])
                punch_info['r_offset'] = r_offset
                punch_info['l_offset'] = rl_offset
                punch_info['r_position'] = (
                    self.r_xavg[0] + r_offset[0], self.r_yavg[0] + r_offset[1], self.r_zavg[0] + r_offset[2])
                punch_info['l_position'] = (
                    self.l_xavg[0] + rl_offset[0], self.l_yavg[0] + rl_offset[1], self.l_zavg[0] + rl_offset[2])
                self.total_r_offset.append(r_offset[4])
                self.total_l_offset.append(rl_offset[4])
            else:
                l_offset = calculate_offset(
                    playerLHandPosLogs, start_time, end_time, sec_num, self.l_xavg[0], self.l_yavg[0], self.l_zavg[0])
                lr_offset = calculate_offset(
                    playerRHandPosLogs, start_time, end_time, sec_num, self.r_xavg[0], self.r_yavg[0], self.r_zavg[0])
                punch_info['l_offset'] = l_offset
                punch_info['r_offset'] = lr_offset
                punch_info['l_position'] = (
                    self.l_xavg[0] + l_offset[0], self.l_yavg[0] + l_offset[1], self.l_zavg[0] + l_offset[2])
                punch_info['r_position'] = (
                    self.r_xavg[0] + lr_offset[0], self.r_yavg[0] + lr_offset[1], self.r_zavg[0] + lr_offset[2])
                self.total_r_offset.append(lr_offset[4])
                self.total_l_offset.append(l_offset[4])

            self.punch_data.append(punch_info)
            # print("所有右拳穩定度:", self.total_r_offset)
            # print("所有右拳穩定度長度:", len(self.total_r_offset))
            # print("所有左拳穩定度:", self.total_l_offset)
            # print("所有左拳穩定度長度:", len(self.total_l_offset))

        self.initialize_guardposition_frame()
        # 初始化繪圖
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.update_plot_and_labels(0)  # 顯示第一拳的數據

    def update_plot_and_labels(self, punch_number):
        punch_info = self.punch_data[punch_number]
        self.ax.cla()

        # 繪製圖表
        self.ax.scatter(self.r_xavg, self.r_zavg, self.r_yavg, color="blue", s=28, label="Initial Rhand Position")
        self.ax.scatter(self.l_xavg, self.l_zavg, self.l_yavg, color="green", s=28, label="Initial Lhand Position")
        self.ax.scatter(self.player_xavg, self.player_zavg, self.player_yavg, color="purple", s=28, label="Head Position")
        
        r_pos = punch_info['r_position']
        l_pos = punch_info['l_position']
        self.ax.scatter(r_pos[0], r_pos[2], r_pos[1], color="darkorange", s=20, label="Rhand Defensive Position")
        self.ax.scatter(l_pos[0], l_pos[2], l_pos[1], color="deeppink", s=20, label="Lhand Defensive Position")
        self.ax.set_title("Defensive Position ")
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Z Axis')
        self.ax.set_zlabel('Y Axis')
        self.ax.legend(loc='upper left', bbox_to_anchor=(0.6, 1), borderaxespad=0, borderpad=0.3, labelspacing=0.2, fontsize=7)
        # self.canvas.draw_idle()


        self.canvas_newboxstance = FigureCanvasTkAgg(self.fig, master=self.canvas)
        self.canvas_newboxstance.draw()
        if hasattr(self, "window_newboxstance"):
            self.canvas.delete(self.window_newboxstance)  # 刪除舊的嵌入窗口
        self.window_newboxstance = self.canvas.create_window(
            740, 390, anchor="nw", window=self.canvas_newboxstance.get_tk_widget(), width=365, height=350
        )
        # 更新標籤
        r_offset = punch_info['r_offset']
        l_offset = punch_info['l_offset']
        # self.frame_righthand_label.config(text=f"Right Hand Move XYZ: ({r_offset[0]}, {r_offset[1]}, {r_offset[2]})")
        # self.frame_righthand_Stability_label.config(text=f"Stability: {r_offset[4]}%")
        # self.frame_lefthand_label.config(text=f"Left Hand Move XYZ: ({l_offset[0]}, {l_offset[1]}, {l_offset[2]})")
        # self.frame_lefthand_Stability_label.config(text=f"Stability: {l_offset[4]}%")
        self.frame_righthand_label.config(text=f"Right Hand:({r_offset[0]}, {r_offset[1]}, {r_offset[2]})")
        self.frame_righthand_Stability_label.config(text=f"Stability:{r_offset[4]}%")
        self.frame_lefthand_label.config(text=f", Left Hand: ({l_offset[0]}, {l_offset[1]}, {l_offset[2]})")
        self.frame_lefthand_Stability_label.config(text=f"Stability: {l_offset[4]}%")

    def on_spinbox_change(self, *args):
        spinbox_value = self.punchnumber_spinbox.get()
        if spinbox_value.isdigit():
            punch_number = int(spinbox_value) - 1
            if 0 <= punch_number < len(self.punch_data):
                self.update_plot_and_labels(punch_number)






    # def initialize_boxstance_frame(self):
    #     if hasattr(self, "boxstance_frame") and self.boxstance_frame is not None:
 
    #         return

    #     self.boxstance_frame = tk.Frame(self.canvas)
        
    #     self.frame_limit_label = tk.Label(self.boxstance_frame, text="Distance:", font=("Arial", 16, "bold"))
    #     self.frame_limit_label.grid(row=0, column=0, padx=1, pady=1)


        
    #     self.boxstance_limit_spinbox_var = tk.StringVar(value="1")
    #     self.boxstance_limit_spinbox = tk.Spinbox(
    #         self.boxstance_frame, from_=1, to=5, textvariable=self.boxstance_limit_spinbox_var, font=("Arial", 16, "bold"), width=2
    #     )
    #     self.boxstance_limit_spinbox.grid(row=0, column=1, padx=1, pady=1)


    def draw_guard_pos(self,ax, filtered_l_xavg, filtered_l_yavg, filtered_l_zavg, filtered_r_xavg, filtered_r_yavg, filtered_r_zavg,after_max_player_z,new_z,new_x,new_y):

        print("進入重新繪畫圖片")
        ax.clear()  # 清空
        ax.scatter(filtered_r_xavg, filtered_r_zavg, filtered_r_yavg, color="blue", s=20, label="Right hand")
        ax.scatter(filtered_l_xavg, filtered_l_zavg, filtered_l_yavg, color="green", s=20, label="Left hand")
        ax.scatter(new_x, new_z, new_y, color="purple", s=40, label="Head")      
        
        def flatten(values):
            if isinstance(values, list):
                return [item for sublist in values for item in (sublist if isinstance(sublist, list) else [sublist])]
            else:
                return [values]
        x_values = flatten(filtered_r_xavg) + flatten(filtered_l_xavg) + flatten([new_x])
        if x_values:
            ax.set_xlim(min(x_values), max(x_values))


        z_values = flatten(filtered_r_zavg) + flatten(filtered_l_zavg) + flatten([new_z])
        if z_values:
            ax.set_ylim(min(z_values), after_max_player_z)

 
        y_values = flatten(filtered_r_yavg) + flatten(filtered_l_yavg) + flatten([new_y])
        if y_values:
            ax.set_zlim(min(y_values), max(y_values))
            
            
            
            
        ax.set_xlim(min(min(filtered_r_xavg), min(filtered_l_xavg)), max(max(filtered_r_xavg), max(filtered_l_xavg), new_x))
        # ax.set_ylim(new_z, after_max_player_z)
        # ax.set_zlim(min(min(filtered_r_yavg), min(filtered_l_yavg), min(new_y)), max(max(filtered_r_yavg), max(filtered_l_yavg), max(new_y)))
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Z Axis')
        ax.set_zlabel('Y Axis')
        ax.set_title("Boxing Stance")
        ax.legend(loc='upper left', bbox_to_anchor=(0.7, 1), borderaxespad=0, borderpad=0.3, labelspacing=0.2, fontsize='small')
        # plt.title("Boxing Stance")
        ax.figure.canvas.draw_idle()
        print("即將結束重新繪畫圖片")
        if hasattr(self, "canvas_boxstance") and self.canvas_boxstance:
            self.canvas_boxstance.draw()
        else:
            self.canvas_boxstance = FigureCanvasTkAgg(ax.figure, master=self.canvas)
            self.canvas_boxstance.draw()
            self.window_boxstance = self.canvas.create_window(
                740, 390, anchor="nw", window=self.canvas_boxstance.get_tk_widget(), width=365, height=350
            )
              



    def show_guardPosLogs(self,file_path): 
        if self.file_path:
            with open(self.file_path,"r",encoding="utf-8") as file:
                data=json.load(file)
                playerRHandPosLogs = data.get("playerRHandPosLogs", [])
                playerLHandPosLogs = data.get("playerLHandPosLogs", [])
                playerPosLogs = data.get("playerPosLogs",[])
                puncherIdx = data.get("puncherIdx", [])
                punchTimeCode = data.get("punchTimeCode", [])
                reactionTime = data.get("reactionTime", [])
                totaltime = math.ceil(punchTimeCode[-1])#這邊來確認總時長是30秒version還是60秒verison
                print(f"totaltime{totaltime}")
                sec_num=len(playerRHandPosLogs)//totaltime
                
                bagPosLogs=data.get("bagPosLogs",[])

                x_bagPos_coords=[point["x"] for point in bagPosLogs]
                y_bagPos_coords=[point["y"] for point in bagPosLogs]
                z_bagPos_coords=[point["z"] for point in bagPosLogs]
                bx=(x_bagPos_coords[0])
                bz=(z_bagPos_coords[0])
                
                rhand,lhand,r_xavg,r_yavg,r_zavg,r_number,r_bag_distance=[],[],[],[],[],[],[]
                for i in range(len(puncherIdx)) :
                    if puncherIdx[i]==0:
                        rhand.append(i) 
                    else:
                        lhand.append(i)
                for i in rhand:
                    # temp=playerRHandPosLogs[:round((punchTimeCode[i]+(reactionTime[i]/2))*sec_num)]
                    temp=playerRHandPosLogs[:round((punchTimeCode[i])*sec_num)]
                    r_number.append(len(temp))
                    x_sum=sum(point["x"] for point in temp)
                    y_sum=sum(point["y"] for point in temp)
                    z_sum=sum(point["z"] for point in temp)
                    
                    r_xavg.append(round(x_sum/len(temp),3))
                    r_yavg.append(round(y_sum/len(temp),3))
                    r_zavg.append(round(z_sum/len(temp),3))
                    distance=round(math.sqrt((round(x_sum/len(temp),3)-bx)**2+(round(z_sum/len(temp),3)-bz)**2),3)
                    r_bag_distance.append(distance)
                # print(f"r_xavg: {r_xavg}") 
                # print(f"r_xavg_number: {len(r_xavg)}")
                # print(f"r_bag_distance: {(r_bag_distance)}")
                l_xavg,l_yavg,l_zavg,l_number=[],[],[],[]
                for i in lhand:
                    temp=playerLHandPosLogs[:round((punchTimeCode[i])*sec_num)]
                    # print(f"punchtimenum{punchTimeCode[i]}")
                    l_number.append(len(temp))
                    x_sum=sum(point["x"] for point in temp)
                    y_sum=sum(point["y"] for point in temp)
                    z_sum=sum(point["z"] for point in temp)
                    
                    l_xavg.append(round(x_sum/len(temp),3))
                    l_yavg.append(round(y_sum/len(temp),3))
                    l_zavg.append(round(z_sum/len(temp),3))
                # print(f"l_xavg: {l_xavg}") 
                # print(f"l_xavg_number: {len(l_xavg)}")

                
                player_xavg,player_yavg,player_zavg,player_number=[],[],[],[]
                for i in range(len(puncherIdx)):
                    temp= playerPosLogs[:round(punchTimeCode[i]*sec_num)]
                    player_number.append(len(temp))
                    x_sum=sum(point["x"] for point in temp)
                    y_sum=sum(point["y"] for point in temp)
                    z_sum=sum(point["z"] for point in temp)
                    
                    player_xavg.append(round(x_sum/len(temp),3))
                    player_yavg.append(round(y_sum/len(temp),3))
                    player_zavg.append(round(z_sum/len(temp),3))
                    
                max_index = player_zavg.index(max(player_zavg))
                new_z = player_zavg[max_index]
                new_x = player_xavg[max_index]
                new_y = player_yavg[max_index]

                fig=plt.figure()
                ax=fig.add_subplot(111,projection='3d')

                print(f"Before r_zavg: {r_zavg}")
                print(f"Before l_zavg: {l_zavg}")
                maxhand_z=max(max(r_zavg),max(l_zavg))
                max_player_z=max(player_zavg)#頭一開始最大位置
                distance=(round((maxhand_z-max_player_z)/5,3))

                def adjust_zavg(max_number, r_xavg, r_yavg, r_zavg, l_xavg, l_yavg, l_zavg, player_zavg, maxhand_z, distance):
                    # maxhand_z=max(max(r_zavg),max(l_zavg))
                    print(f"maxhand_z{maxhand_z}")               
                    max_num=int(max_number)
                    max_player_z=max(player_zavg)
                    print(f"max_player_z{max_player_z}")            
                    # distance=round((maxhand_z-max_player_z)/5,2)
                    print(f"distance{distance}")
                    after_max_player_z=max_player_z
                    for _ in range(max_num):
                         after_max_player_z+=distance
                    print(f"After max_player_z{after_max_player_z}")
                    
                    
                    filtered_r_xavg = []
                    filtered_r_yavg = []
                    filtered_r_zavg = []

                    for x, y, z in zip(r_xavg, r_yavg, r_zavg):
                        if z > max_player_z and z <after_max_player_z: 
                            filtered_r_xavg.append(x)
                            filtered_r_yavg.append(y)
                            filtered_r_zavg.append(z)
                            
                    if not filtered_r_xavg:
                        filtered_r_xavg = [0]
                    if not filtered_r_yavg:
                        filtered_r_yavg = [0]
                    if not filtered_r_zavg:
                        filtered_r_zavg = [0]

                    filtered_l_xavg = []
                    filtered_l_yavg = []
                    filtered_l_zavg = []
                    for x, y, z in zip(l_xavg, l_yavg, l_zavg):
                        if z > max_player_z and z <after_max_player_z:  
                            filtered_l_xavg.append(x)
                            filtered_l_yavg.append(y)
                            filtered_l_zavg.append(z)
                    if not filtered_l_xavg:
                        filtered_l_xavg = [0]
                    if not filtered_l_yavg:
                        filtered_l_yavg = [0]
                    if not filtered_l_zavg:
                        filtered_l_zavg = [0]
                    # return filtered_l_xavg, filtered_l_yavg, filtered_l_zavg, filtered_r_xavg, filtered_r_yavg, filtered_r_zavg,after_max_player_z
                    return (
            filtered_l_xavg, filtered_l_yavg, filtered_l_zavg, 
            filtered_r_xavg, filtered_r_yavg, filtered_r_zavg, 
            after_max_player_z
    )

                adjust_zavg(1, r_xavg, r_yavg, r_zavg, l_xavg, l_yavg, l_zavg, player_zavg, maxhand_z, distance)
                filtered_l_xavg, filtered_l_yavg, filtered_l_zavg, \
                filtered_r_xavg, filtered_r_yavg, filtered_r_zavg, after_max_player_z = adjust_zavg(
                    1, r_xavg, r_yavg, r_zavg, l_xavg, l_yavg, l_zavg, player_zavg, maxhand_z, distance
                )
                self.draw_guard_pos(ax, filtered_l_xavg, filtered_l_yavg, filtered_l_zavg, filtered_r_xavg, filtered_r_yavg, filtered_r_zavg,after_max_player_z,new_z,new_x,new_y)
                print(f"Filtered r_zavg: {r_zavg}")
                print(f"Filtered l_zavg: {l_zavg}")
                self.initialize_boxstance_frame()
                

            def on_spinbox_change(*args):
                
                spinbox_range_value = self.boxstance_limit_spinbox.get()
                print(f"spinbox_boxstance_value={spinbox_range_value}")

                if spinbox_range_value.isdigit():
                    max_number = int(spinbox_range_value)  
                    filtered_l_xavg, filtered_l_yavg, filtered_l_zavg, \
                    filtered_r_xavg, filtered_r_yavg, filtered_r_zavg, after_max_player_z = adjust_zavg(
                        max_number, r_xavg, r_yavg, r_zavg, l_xavg, l_yavg, l_zavg, player_zavg, maxhand_z, distance
                    )
                    self.draw_guard_pos(
                        ax, filtered_l_xavg, filtered_l_yavg, filtered_l_zavg,
                        filtered_r_xavg, filtered_r_yavg, filtered_r_zavg,
                        after_max_player_z, new_z, new_x, new_y
                    )
                    fig.canvas.draw_idle()
                    print(f"The value updated to: {max_number}")             
                    
                    


            self.boxstance_limit_spinbox_var.trace_add("write", lambda *args: on_spinbox_change())
            self.canvas.create_window(900, 700, anchor="nw", window=self.boxstance_frame)

            # 使用 FigureCanvasTkAgg 將圖表嵌入到 Tkinter
            self.boxstance_frame.place(x=780, y=743) 
            # self.canvas_boxstance = FigureCanvasTkAgg(fig, master=self.canvas)
            # self.canvas_boxstance.draw()
            # self.window_boxstance=self.canvas.create_window(740, 390, anchor="nw", window=self.canvas_boxstance.get_tk_widget(), width=365, height=350)
            # self.boxstance_frame.place(x=740, y=743)        
        
    def data_normalization(self,file_path):
        if self.file_path:
            with open(self.file_path,"r",encoding="utf-8") as file:
                global percentage_series 
                data=json.load(file)
                playerRHandPosLogs = data.get("playerRHandPosLogs", [])
                playerLHandPosLogs = data.get("playerLHandPosLogs", [])
                playerPosLogs = data.get("playerPosLogs",[])
                puncherIdx = data.get("puncherIdx", [])
                punchTimeCode = data.get("punchTimeCode", [])
                reactionTime = data.get("reactionTime", [])
                punchspeed = data.get("punchSpeed", [])
                punchPower = data.get("punchPower", [])
                totaltime = math.ceil(punchTimeCode[-1])#這邊來確認總時長是30秒version還是60秒verison
                

                
                if totaltime not in [30, 60]:
                    if abs(60-totaltime)<abs(30-totaltime):
                        totaltime=60
                    else:
                        totaltime=30
                sec_num=len(playerRHandPosLogs)//totaltime
                r_first=0
                l_first=0
                rhand,lhand,r_xavg,r_yavg,r_zavg,r_number,r_bag_distance=[],[],[],[],[],[],[]
                l_xavg,l_yavg,l_zavg,l_number=[],[],[],[]
                for i in range(len(puncherIdx)) :
                    if puncherIdx[i]==0:
                        rhand.append(i) 
                    else:
                        lhand.append(i)
                for i in range(len(puncherIdx)):
                    if puncherIdx[i] ==0:
                # for i in rhand:
                    # temp=playerRHandPosLogs[:round((punchTimeCode[i]+(reactionTime[i]/2))*sec_num)]
                        end_index = round(punchTimeCode[i] * sec_num)
                        if end_index > len(playerRHandPosLogs):
                            end_index = len(playerRHandPosLogs)

                        temp_r = playerRHandPosLogs[r_first:end_index]
                        temp_l = playerLHandPosLogs[r_first:end_index]                        
                        r_first = end_index  # 更新起始索引
                        r_number.append(len(temp_r))
                        x_sum=sum(point["x"] for point in temp_r)
                        y_sum=sum(point["y"] for point in temp_r)
                        z_sum=sum(point["z"] for point in temp_r)
                        
                        r_xavg.append(round(x_sum/len(temp_r),3))
                        r_yavg.append(round(y_sum/len(temp_r),3))
                        r_zavg.append(round(z_sum/len(temp_r),3))
                        
                        
                        x_sum=sum(point["x"] for point in temp_l)
                        y_sum=sum(point["y"] for point in temp_l)
                        z_sum=sum(point["z"] for point in temp_l)
                        
                        l_xavg.append(round(x_sum/len(temp_l),3))
                        l_yavg.append(round(y_sum/len(temp_l),3))
                        l_zavg.append(round(z_sum/len(temp_l),3))
                # print(f"normaliza_r_xavg,len(r_xavg){r_xavg},{len(r_xavg)}")
                    else:

                # for i in lhand:
                        end_index = round(punchTimeCode[i] * sec_num)
                        if end_index > len(playerRHandPosLogs):
                            end_index = len(playerRHandPosLogs)
                        # temp=playerLHandPosLogs[:round((punchTimeCode[i])*sec_num)]
                        # temp=playerLHandPosLogs[l_first:end_index]
                        temp_r = playerRHandPosLogs[l_first:end_index]
                        temp_l = playerLHandPosLogs[l_first:end_index] 
                        l_first = end_index 
                        l_number.append(len(temp_l))
                        x_sum=sum(point["x"] for point in temp_l)
                        y_sum=sum(point["y"] for point in temp_l)
                        z_sum=sum(point["z"] for point in temp_l)
                        
                        l_xavg.append(round(x_sum/len(temp_l),3))
                        l_yavg.append(round(y_sum/len(temp_l),3))
                        l_zavg.append(round(z_sum/len(temp_l),3))
                        
                        x_sum=sum(point["x"] for point in temp_r)
                        y_sum=sum(point["y"] for point in temp_r)
                        z_sum=sum(point["z"] for point in temp_r)
                        
                        r_xavg.append(round(x_sum/len(temp_r),3))
                        r_yavg.append(round(y_sum/len(temp_r),3))
                        r_zavg.append(round(z_sum/len(temp_r),3))
                print(f"normaliza_l_xavg,len(l_xavg){l_xavg},{len(l_xavg)}")    
                print(f"normaliza_r_xavg,len(r_xavg){r_xavg},{len(r_xavg)}")
            # print("所有右拳穩定度:", self.total_r_offset)
            # print("所有右拳穩定度長度:", len(self.total_r_offset))
            # print("所有左拳穩定度:", self.total_l_offset)
            # print("所有左拳穩定度長度:", len(self.total_l_offset))
         
                def normalize_features(features):
                    scaler = MinMaxScaler()
                    return scaler.fit_transform(np.array(features).reshape(-1,1)).flatten()            
                if self.percentage_series is None:
                    raise ValueError("self.percentage_series 未被正確設置，請確保調用了生成 percentage_series 的方法。")
            
                normalize_reactionTime = normalize_features(reactionTime)
                normalize_punchspeed = normalize_features(punchspeed)
                normalize_punchPower = normalize_features(punchPower)
                normalize_l_xavg = normalize_features(l_xavg)
                normalize_l_yavg = normalize_features(l_yavg)
                normalize_l_zavg = normalize_features(l_zavg)
                normalize_r_xavg = normalize_features(r_xavg)
                normalize_r_yavg = normalize_features(r_yavg)
                normalize_r_zavg = normalize_features(r_zavg)
                normalize_total_r_offset = normalize_features(self.total_r_offset)   
                normalize_total_l_offset = normalize_features(self.total_l_offset)
                    
                features={
                "Percentage Series" :self.percentage_series.to_dict(),#
                
                "normalize_punchspeed": normalize_punchspeed.tolist(),
                "normalize_punchPower": normalize_punchPower.tolist(),
                "normalize_lhand_xaverge": normalize_l_xavg.tolist(),
                "normalize_lhand_yaverge": normalize_l_yavg.tolist(),
                "normalize_lhand_zaverge": normalize_l_zavg.tolist(),
                "normalize_rhand_xaverge": normalize_r_xavg.tolist(),
                "normalize_rhand_yaverge": normalize_r_yavg.tolist(),
                "normalize_rhand_zaverge": normalize_r_zavg.tolist(),
                "normalize_total_rhand_stability": normalize_total_r_offset.tolist(),
                "normalize_total_lhand_stability": normalize_total_l_offset.tolist()
                }
                # print(features)
                print("Percentage Series:", self.percentage_series)
                return features
    
    def send_data_to_gemini(self, normalized_features):
        normalized_features_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in normalized_features.items()
        }

        prompt = (
            "You are a Professional Boxing Coach and Professional Data Analyst.\n"
            "Based on the following normalized feature data and summary percentage data, "
            "please act as a professional boxing coach with expertise in analyzing boxing performance. "
            "The Percentage Series represents the percentile rank compared to all other data, "
            "indicating the percentage of performance this user has surpassed. "
            "The remaining data under formative data reflects key metrics normalized for analytical purposes. "
            "Your task is to provide: "
            "First, a creative and descriptive name for the boxing style that fits the user's performance metrics. "
            "Second, a thoughtful and motivating sentence that encourages the user to reflect on their performance, "
            "highlighting strengths and areas for improvement, and helping them quickly and comprehensively understand "
            "their data while feeling a sense of recognition and alignment with the analysis. "
            "Third, point out the current numerical weaknesses in not only the summary data in the Percentage Series "
            "but also the normalized two-handed stability can be used as part of the weaknesses, to pinpoint which "
            "stability sections are susceptible to instability before, during and after the time period. "
            "That should be addressed to have a better boxing performance, suggesting reasonable ways to strengthen them, "
            "such as which values to improve the basic boxing movements or certain parts of the muscle training that can be very helpful "
            "and improve the weaknesses. Title only needs stat at Boxing Style: and Personal Advantages: and Weaknesses to Address: "
            "and no additional punctuation should be added to the three titles mentioned earlier. "
            "Make sure all feedback text sounds a natural easy-going and conversational tone, response start with Boxing style:"
            "and no extra punctuation around titles. "
            "Output must be simple text, each section separated by a blank line."
            "Please make sure the output contains only plain text with no Markdown, no asterisks (*), no hashes (#)"
            "\n\nJSON:\n"
            f"{json.dumps(normalized_features_serializable, ensure_ascii=False, indent=2)}"
        )

        try:
            model = genai.GenerativeModel("models/gemini-2.0-flash")  
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.6,
                    "max_output_tokens": 250,  
                    "top_p": 0.95,
                    "top_k": 40
                }
            
            )
            if response.parts:
                result_text = "".join([p.text for p in response.parts if hasattr(p, "text")])
            else:
                print(" No content returned.")
                result_text = ""
            return result_text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None
                
    def showformativedata(self):
        if self.file_path:
            self.clear_canvas_content()
            initial_memory = self.get_memory_usage()
            print(f"Animation Initial memory usage: {initial_memory:.2f} MB")
            
            
            
            
            # img= Image.open("show_plot_punch_and_reaction.png")
            # img = img.resize((1460,300),Image.LANCZOS)
            # photo= ImageTk.PhotoImage(img)
            # self.canvas.create_image(10, 90, anchor="nw", image=photo)#圖片位置
            # self.photo_images=[photo]

            
            
            self.show_plot_punch_and_reaction(self.file_path)
            self.show_boxingstance(self.file_path)
            
            # self.show_punch_position(self.file_path)#全部確切的拳擊
            self.show_all_position(self.file_path)
            self.show_Boxing_Stance_scatter(self.file_path)
            self.show_animation()
            # self.show_guardPosLogs(self.file_path)


            # normalized_features = self.data_normalization(self.file_path)
            # gpt_result = self.send_data_to_gpt(normalized_features)
            # if gpt_result:
            #     # 在 Tkinter 畫布上顯示 GPT 結果
            #     result_label = tk.Label(
            #         self.canvas, 
            #         text=f"GPT Result: {gpt_result}", 
            #         font=("Arial", 12), 
            #         wraplength=600  # 控制文本換行
            #     )
            #     result_label.place(x=50, y=500)  # 自定義位置
            #     print("GPT 分析結果：")
            #     print(gpt_result)
                
            final_memory = self.get_memory_usage()
            print(f"Final memory usage: {final_memory:.2f} MB")
            
    def showboxingstyle(self):
        if self.file_path:
            self.clear_canvas_content()
            initial_memory = self.get_memory_usage()
            print(f"Animation Initial memory usage: {initial_memory:.2f} MB")
            
            # self.test_label=tk.Label(self.canvas,text="test correct",font=("Arial", 20))
            # self.test_label.place(x=(self.window_width//2-50),y=self.window_height//2-50)
            
            normalized_features = self.data_normalization(self.file_path)
            gpt_result = self.send_data_to_gemini(normalized_features)   #抓取gemini 結果
            if gpt_result:
                # 在 Tkinter 畫布上顯示 GPT 結果
                self.test_label = tk.Label(
                    self.canvas, 
                    # text=f"GPT Result: {gpt_result}", 
                    text=gpt_result,
                    font=("Arial", 12), 
                    wraplength=600  # 控制文字換行
                )
                self.test_label.place(x=self.window_width//2-280, y=self.window_height//2-180)# 自定義位置
                print("GPT 分析結果：")
                print(gpt_result)
                
                
                
            final_memory = self.get_memory_usage()
            print(f"Final memory usage: {final_memory:.2f} MB")
            
if __name__=='__main__':
    app=App()
    app.mainloop()
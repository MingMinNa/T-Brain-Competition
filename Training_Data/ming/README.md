# Training Data (ming) 說明

## 資料夾整體結構

```text
Training_Data/
├─ Generate/                         # 自行產生與處理後的資料
│  ├─ 36_Merged_TrainingData/        # 一分鐘資料（合併後）
│  ├─ Additional_Features/           # 由中央氣象局爬取的天文資料
│  └─ Average_Data/                  # 十分鐘平均後的資料
│     ├─ Train(AVG)/                 # 僅保留資料完整的十分鐘
│     └─ Train(IncompleteAVG)/       # 允許資料不完整的十分鐘
│
└─ Given/                            # 競賽方提供的原始資料
   ├─ 36_TrainingData/               # 一分鐘資料
   ├─ 36_TrainingData_Additional_V2/ # 一分鐘補充資料
   └─ Average_Data/                  # 官方提供的十分鐘資料（範例）
```

## 資料來源分類說明

### `Given/`（競賽方提供）
此資料夾內的所有內容均未經自行處理，屬於官方或原始資料。

#### `36_TrainingData/`
- 各地點的一分鐘解析度資料
- 為主要的原始訓練資料來源

#### `36_TrainingData_Additional_V2/`
- 與 `36_TrainingData` 格式相同
- 作為補充用的一分鐘資料（時間或樣本補齊）

#### `Average_Data/`
- 由競賽範例程式中擷取的十分鐘資料
- 僅涵蓋 09:00 ~ 16:50 的時間範圍
- 未用於主要訓練流程，主要作為參考或對照

---

### `Generate/`（自行生成）
此資料夾包含前處理後的資料。

#### `36_Merged_TrainingData/`
將 `36_TrainingData` 和 `36_TrainingData_Additional_V2` 兩份資料合併後產生
- 資料解析度：一分鐘
- 作為後續特徵工程與平均處理的基礎資料

#### `Additional_Features/`
由 [中央氣象局](https://www.cwa.gov.tw/V8/C/K/astronomy_day.html) 網站額外爬取 （太陽仰角、太陽方位角）
- 資料解析度：十分鐘
- 屬於後期加入的額外特徵，用於輔助模型訓練

#### `Average_Data/`
由 `36_Merged_TrainingData` 搭配 `Additional_Features` 產生的十分鐘資料
- 以「十分鐘」為一個時間區間
- 將該區間內的一分鐘資料取平均，作為該十分鐘的代表值

#### `Train(AVG)/`
僅保留資料完整的十分鐘
- 條件：該十分鐘內 10 分鐘皆有資料
- 資料品質最佳，為主要使用的訓練資料

#### `Train(IncompleteAVG)/`
允許資料不完整
- 即使該十分鐘內只有部分分鐘有資料，仍會取平均後保留
- 資料品質較不穩定，實際使用率較低


#### 範例說明
| 時間區間        | 有資料的分鐘    | Train(AVG) | Train(IncompleteAVG) |
| ----------- | --------- | ---------- | -------------------- |
| 10:00–10:09 | 10 分鐘皆有   | ✓ 納入       | ✓ 納入                 |
| 10:00–10:09 | 僅 03、05 分 | ✗ 排除       | ✓ 納入（取 2 分鐘平均）       |

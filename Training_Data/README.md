## Training Data

### 資料夾介紹
* ``Generate``
    均為自行生成的
    * ``36_Merged_TrainingData``  
        結合 ``36_TrainingData`` 和 ``36_TrainingData_Additional_V2``，為各個地點以一分鐘為單位的資料。
    * ``Additional_Features``  
        為從[中央氣象局](https://www.cwa.gov.tw/V8/C/K/astronomy_day.html)額外爬取的資料，以十分鐘為單位的太陽仰角與方位角。
    * ``Average_Data``  
        結合 ``36_Merged_TrainingData`` 以及 ``Additional_Features`` 產生。  
        計算方式大致為 **該十分鐘內的資料算平均，當作這十分鐘的資料**。  
        依照資料完整性，又可分成下列兩個資料夾
        * ``Train(AVG)``  
            若該十分鐘內均有資料，則將其算平均後，納入此資料內。  
        * ``Train(IncompleteAVG)``  
            不管該十分鐘內是否均有資料，將該十分鐘內的資料算平均後，納入此資料內。

        舉例來說：  
        1. 假設在 ``10:00 ~ 10:09`` 均有資料，則 ``Train(AVG)`` 和 ``Train(IncompleteAVG)`` 均有這十分鐘平均的資料。
        2. 假設在 ``10:00 ~ 10:09`` 中，只有 ``03``, ``05``分有資料，則 ``Train(AVG)`` 不會記錄這筆資料，但 ``Train(IncompleteAVG)`` 則會將這兩分鐘算平均，作為這十分鐘平均的資料。

    註：
    * ``Additional_Features`` 是後來加上的資料，且 ``Train(IncompleteAVG)`` 資料比較沒用到。因此，並未將 ``Additional_Features`` 合併進 ``Train(IncompleteAVG)`` 內。
    * 在 ``TrainData(AVG)`` 中，有一個稱為 ``Power(mW)_1`` 的欄位，這是在測試**將前十分鐘的結果納入輸入特徵**時，所特地加上的欄位，忽略即可。
* ``Given``
    均由競賽方提供的資料
    * ``36_TrainingData``  
        為各個地點以一分鐘為單位的資料。
    * ``36_TrainingData_Additional_V2``  
        內容與 ``36_TrainingData`` 相似，是其補充資料。
    * ``Average_Data``  
        從範例程式中的資料擷取下來，以十分鐘為單位的資料，時間範圍限定在 ``9:00 ~ 16:50``。
    

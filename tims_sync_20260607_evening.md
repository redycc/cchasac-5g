# sync #1 — Tim 這邊 C-HASAC 線

**撰寫時間:2026-06-07(日)20:56 CST。** 內文引用的群裡討論,涵蓋到今晚 ~20:33 的訊息
(z1/z0 live 監控、base-first vs z-first 討論、critic 不吃 z 的說明)。

> **一句話 TL;DR:我這邊的資料顯示「z 編好了」和「z 被用」是兩件事 —— pure-RL 時 policy 不用 z、
> BC 進來才打開「用」的開關。所以「先穩 base」和「讓 +z 爬更高」可能不必二選一:兩個軸可以
> 同時監測(方法見 §1d)。細節如下,趕時間看完 §1 就可以。**

這是我這線第一次把結果整理過來(之前只有讀群裡的訊息,沒貼過東西),所以一次講完整一點。

setup 差異先說:我這邊是 3/6-cell、4 RB 的 **sum-rate snapshot 環境**,SAC + context encoder(z),
大部分量測是 **pure-RL(無 BC)、N_UE=10/20**(對齊成你們的 N_UE=30/60 重跑中)——
跟你們的 logpf+BC / PF-utility 不同,**絕對數字不可直接比**;可比的是機制層面。

名詞先對齊(下面會一直用):
- **z-probe**:eval 時把 z 換成 z←0 或 z←shuffle 再量性能 —— 你們已經在做 z←0,
  z←shuffle 只是換一種替換(把 z 換成同 batch 裡別的 episode 的 z)。
- **drop_shuffle** = z 被 shuffle 後性能掉多少,**> 0 才代表 policy 真的在用 z 的「內容」**。
  ⚠️ 它跟你們常報的 z←0(drop_zero)**不直接可比**:shuffle 比 zero 嚴格(保留了 z 的邊際
  分布,只破壞對應關係),所以你們 v16 的 drop_zero +11.87 不能直接對到這欄。
- 我們的數字都是 ≥10 seed、訓練末態的 median[IQR],train/eval 的環境 seed 完全分開。

---

## §1 跟你們 z1/z0 討論最相關的量測:encode vs use

看到群裡在討論「先讓大家都更好,還是讓 +z 爬更高」,我這邊有一組資料也許可以參考:

**(a) z 裡面有沒有協調資訊?有,而且 pure-RL 就有。**
把訓練好的 z 凍住,用一個線性分類器從 z 去 decode「鄰近 BS 在各 RB 有沒有開功率」
(3-cell·N_UE=10):**AUC=0.936**;對照組 —— 只用自家 BS 的本地觀測去 decode,只有 0.641。
→ z 確實多帶了不少鄰居的協調資訊。

**(b) 但 policy 用不用它?pure-RL 不用;+BC 才開始用。**(同一組 setup,3-cell·80k)
(表格手機上可能跑版,重點一句:**pure-RL 的 drop_shuffle = 0.00,+BC 之後全部 > 0**)

| variant | ±BC | sum-rate med | drop_shuffle |
|---|---|--:|--:|
| per-BS z(= 你們 v16 那條路)| pure-RL | 74.4 | **0.00** |
| per-BS z | +BC | 68.8 | 1.87 |
| per-BS z + 拿掉 own-KPM | +BC | 71.3 | **2.75** |
| broadcast z | +BC | 70.6 | **4.64** |
| flat,z≡0(≈ 你們的 z0)| +BC | 62.4 | — |

(own-KPM = obs 裡「自家 BS 自己的 KPI 歷史」那一段;對應你們 obs 的話,大概是 local
觀測裡 serving-BS 自己的那幾維,不含 z。)

**(c) 我們目前的讀法:**「z 有沒有編好」和「z 有沒有被用」是兩個軸 —— 協調資訊一直在 z 裡(a),
但 pure-RL 的 policy 不去動它,BC 進來才翻開「用」的開關(b)。
你們現在 z1(logpf+BC)看到 z←0 會掉、z0 追不上 —— 跟這個讀法一致,可以互相印證。
對「先 base 還是先 z」的隱含意義(僅觀察、非定論):**base 的強度跟 z 有沒有被用,看起來是
部分解耦的** —— 墊高 base 不一定自動帶動 z 被用;z 被不被用,比較跟著「有沒有直接作用在
actor 上的介入(像 BC)」走。

**(d) 跟「先穩 base 還是推 z」可能直接相關的一筆:** 我們試過在 z 上加對比學習正則
(InfoNCE,細節不重要)掃了一輪(3-cell·80k):唯一讓 z 被大量使用的設定(drop_shuffle
中位數 **+20.6**)同時把 sum-rate 中位數從 86.4 拉到 **58.5**;而所有穩定的設定,
drop_shuffle 全部 ≈0。在我們這邊,「變穩」和「z 被用」**還沒有同時拿到過**。
→ 如果你們走「先讓大家都更好」,也許值得在穩 base 的同時順手多量一個 z←shuffle
(你們的 z←0 流程改一行就有),避免 z-usage 默默歸零而沒被發現。
另外「z 有沒有編好」不用等 base 變強才能判斷 —— (a) 那個線性 probe 可以直接量;
**probe 的程式碼我整理好就貼過來**(很短,單檔)。這樣 base-first / z-first 就不一定是
先後抉擇,兩個軸可以同時被監測。

**(e) 一個成本不高、可以在你們現在 run 上試的旋鈕:** 上表「穩定且 z 有被用」最乾淨的一筆是
**把 own-KPM 從 obs 拿掉 + BC**(z 變成唯一的跨 BS 資訊通道,drop_shuffle 2.75、sum 還比
不拿掉高)。類似的 ablation(把 obs 裡跟 z 重複的那一路拿掉)或許值得在 logpf+BC 上跑一次看看。

*(§2 起是延伸/印證,趕時間可以先略過。)*

## §2 critic 端介入 —— 我們這邊排除掉的一個方向

看到群裡最新的說明:你們現在的 critic 是**故意不吃 z** 的(只吃 share_obs + 全動作)。
我們這邊有個實驗結果正好跟這個設計選擇相關 —— 我們試了另一個極端:
**讓中央 critic 的 state 只走 z**(z 成為 critic 唯一的跨 BS 通道;3-cell·80k·10seed·N_UE=10):

- critic 端確實能依賴 z、sum-rate 也有小幅提升,但 **actor 的 drop_shuffle 還是 ≈ 0**。
- 讀法(僅觀察):至少在我們這邊,就算把 critic 完全壓在 z 上,actor 也不會因此去用 z ——
  「z 沒被用」的瓶頸**不在 critic 端**,比較像在 shared actor。
  → 你們「critic 不吃 z」的設計跟這個觀察是相容的;想撬動 z-usage 的話,critic 側
  大概不是關鍵,直接作用在 actor 上的介入(像 BC,§1)比較有機會。

## §3 flat 路線 —— 跟你們的結論互相印證

我們這邊也試過直接救 flat(z≡0):batch 加到 1024 確實把 seed 間變異壓小(IQR 7.4),
但整體水準停在低檔(median 39.5;好的 seed 其實摸得到 86+,只是大多數卡在低檔);
把 warmup 拉長幾檔也一樣上不去。
→ 在我們試過的設定裡,flat 連續功率 + SAC 沒有學到協調,單純調穩定性的旋鈕沒把它救起來。
這跟你們固定功率 −4.17 追平 RL(−4.26)那條證據鏈、還有「轉 explicit-opt / imitation」的
討論方向一致 —— 兩邊各自獨立走到同一個結論,報告裡可以互相印證。
順帶一提:我們 +BC 的結果(§1b)從機制面看也算 imitation 路線的正面證據 ——
supervised 信號進來之後,不只沒崩,z 才開始被用。

## §4 評估方式 —— 我們踩過的坑,分享一下

我們這邊看過同一個設定、不同 seed 的末態從 26 跑到 86(雙峰分布),所以一律
**≥10 seed、報訓練末態的 median[IQR]、train/eval 環境 seed 分開**,結論才穩得下來。
你們 z1 現在 best −0.317、同段也有 −11.8 的震幅 —— 群裡好像也有提到要確認這個 peak
可不可重複,方向正好一致:多幾個 seed 把它 median 化之後,「z1 > z0」會更站得住。
**需要的話,我們跑多 seed + 評估(含 z-probe)的 script 可以整包給你們**(說一聲就貼)。

## §5 我們線正在跑的(供進度同步,不需要你們動作)

- **scale 的早期訊號**:6-cell(N_UE=20·200k)broadcast pure-RL 有 **2/6 個 seed** 出現真的
  inference-time z(drop_shuffle +12…+20)—— 不穩、變異大,但這是 pure-RL 下唯一看到 z 被用的
  條件,所以我們在押「放大 N + 拉長訓練」。
- 主要實驗已對齊你們的 **N_UE=30/60** 重跑中(3-cell·80k·10seed 六個變體 + 6-cell·400k 的
  2-seed 先導),收完會有對齊設定的 critic-端 / scale 數字,到時再貼第二篇。

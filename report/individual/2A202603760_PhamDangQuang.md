# Báo cáo Day 7 — Agentic RAG

**Họ và tên:** Phạm Đăng Quang  
**Mã số sinh viên (MSSV):** 2A202603760  
**Lớp:** K5-L3B  
**Repo bài nộp:** `K5-L3B-DAY07-2A202603760-PhamDangQuang-RetrievalContext`

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - *Tài liệu nguồn*: NIST AI Risk Management Framework 1.0 (NIST Special Publication AI 100-1), xuất bản tháng 1/2023. Link chính thức: `https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf`.
  - *Bản quyền*: Public Domain (Tác phẩm của cơ quan chính phủ Hoa Kỳ, tự do phân phối và nghiên cứu phi thương mại/thương mại).
  - *Số trang*: 48 trang vật lý (43 trang nội dung hữu ích từ trang vật lý 5 đến 47, loại trừ trang bìa, mục lục và thông cáo phát hành; vượt yêu cầu tối thiểu $\ge 30$ trang).
  - *Số chunk kế thừa từ Day 6*: **159 chunks** được phân đoạn theo cấu trúc ngữ nghĩa/đoạn văn (`collection_strategy_b` trong Qdrant local, trần kích thước $\le 800$ ký tự để bảo toàn toàn vẹn ngữ cảnh).
  - *Định dạng chunk*: Lưu tại `data/chunks.json`, bao gồm các trường chuẩn: `id`, `source_id` ("nist_ai_rmf_1_0"), `source` ("data/NIST_AI_100_1.pdf"), `page` (số trang vật lý), `text`, và `metadata` (`section`, `part`, `chunk_size`, `strategy`).

- **Vector store, embedding model, phiên bản và số chiều:**
  - *Vector store*: `Qdrant` (thư viện `qdrant-client` v1.20.0), lưu trữ cục bộ file-based tại `../K5-L3B-DAY06-2A202603760-DataIngestion/artifacts/qdrant_db`, sử dụng độ đo Cosine Distance (`Distance.COSINE`).
  - *Embedding model*: `BAAI/bge-small-en-v1.5` thông qua thư viện `fastembed` (v0.9.1), chạy local ONNX Runtime tối ưu CPU, 384 chiều (`dimensions = 384`), chi phí truy vấn $0.00.

- **Agent model, cách chạy và phân chia dev/test:**
  - *Agent model*: `openai/gpt-4o-mini` (thông qua OpenRouter API / OpenAI API client), cấu hình `temperature=0.0` và `response_format={"type": "json_object"}` để đảm bảo tính tất định và chuẩn hóa cấu trúc dữ liệu tool-calling.
  - *Khung điều phối*: Kiến trúc Agent Controller đa lượt (`run_agent` tại `src/agent.py`), tối đa 4 lượt truy xuất (`max_searches = 4`), context budget 1600 ký tự.
  - *Phân chia Benchmark*: Tổng cộng **24 câu hỏi** lưu tại `data/agent_cases.json`:
    - **Dev split (17 câu)**: 15 câu có đáp án (`q01`–`q15`) + 2 câu thiếu evidence (`q_unans_dev_01`, `q_unans_dev_02`).
    - **Test split (7 câu)**: 5 câu có đáp án (`q16`–`q20`) + 2 câu thiếu evidence (`q_unans_test_01`, `q_unans_test_02`).

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án*: Kế thừa từ benchmark Day 6 (`benchmark.json`), mỗi câu gồm `id`, `question`, `expected_answer`, và danh sách `gold` gồm `source_id`, `page` và `evidence` (chuỗi trích dẫn nguyên văn `evidence_quote`). Đã kiểm tra tự động bằng `starter/baseline.py`, 100% (20/20 câu) có evidence nằm trọn vẹn trong chunk đúng nguồn và số trang.
  - *Câu thiếu evidence*: Các câu hỏi được thiết kế gần chủ đề quản trị rủi ro AI nhưng cố tình không tồn tại bằng chứng trong văn bản NIST AI RMF 1.0 (ví dụ: mức phạt tiền cụ thể, ngưỡng năng lực tính toán FLOPs của foundation model, thủ tục cấp giấy phép bay thương mại của drone, hay điểm benchmark MMLU tối thiểu). Tiêu chí kiểm chứng: duyệt toàn văn tài liệu để khẳng định tài liệu không chứa các quy định bắt buộc/mức phạt mang tính cưỡng chế pháp lý này.
  - *Tiêu chí chấm*:
    - **Retrieval metrics**: Hit@1, Evidence Hit@3 (có chunk chứa gold evidence trong top 1 hoặc 3), Context Evidence Rate (tỷ lệ context cuối cùng nạp vào LLM chứa đủ evidence).
    - **Agent metrics**: Tool use rate (gọi ít nhất 1 lần retrieval), Grounded answer (thang điểm 0/1/2 theo mức độ chính xác và trích dẫn), Citation validity (tất cả citation ID phải có trong context và hỗ trợ câu trả lời), Abstention (từ chối chính xác câu thiếu evidence và không từ chối nhầm câu có đáp án).

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | TF-IDF lexical, top-k=3, no filter, budget=800 | Baseline từ vựng thuần túy offline, không dùng embedding | dev | 12/15 (80.0%) | 12/15 (80.0%) | 8/15 (53.3%) | 1.5 ms / $0.00 |
| **Phương pháp 2** | Dense (Qdrant + BGE-Small), top-k=3, no filter, budget=800 | Tìm kiếm vector ngữ nghĩa giúp bắt các câu hỏi diễn đạt lại | dev | 10/15 (66.7%) | 13/15 (86.7%) | 7/15 (46.7%) | 24.5 ms / $0.00 |
| **Phương pháp 3** | Hybrid RRF (Dense + BM25, k=60), top-k=3, no filter, budget=800 | Kết hợp điểm mạnh từ vựng (từ khóa kỹ thuật) và ngữ nghĩa | dev | 11/15 (73.3%) | 14/15 (93.3%) | 8/15 (53.3%) | 32.0 ms / $0.00 |
| **Cải thiện 1** | Dense + Metadata Filter (`section`), top-k=3, budget=800 | Bộ lọc metadata theo phân vùng tài liệu loại trừ nhiễu | dev | 10/15 (66.7%) | 13/15 (86.7%) | 7/15 (46.7%) | 27.1 ms / $0.00 |
| **Cải thiện 2** | Hybrid RRF + Budget 1600 ký tự, top-k=3, no filter | Tăng context budget từ 800 lên 1600 ký tự để chứa đủ $\ge 2$ chunks | dev | 11/15 (73.3%) | 14/15 (93.3%) | **13/15 (86.7%)** | 29.7 ms / $0.00 |
| **Cấu hình đã chọn** | Hybrid RRF + Budget 1600 ký tự, top-k=3, no filter | Cấu hình tối ưu chốt từ Dev được đánh giá độc lập trên Test | test | 4/5 (80.0%) | 4/5 (80.0%) | **4/5 (80.0%)** | 33.9 ms / $0.00 |
| **Baseline (Test)** | TF-IDF lexical, top-k=3, no filter, budget=800 | Đối chứng baseline trên tập Test | test | 4/5 (80.0%) | 5/5 (100.0%) | 3/5 (60.0%) | 1.8 ms / $0.00 |

*Ghi chú:* Toàn bộ các thử nghiệm retrieval chạy offline trên CPU máy tính cá nhân qua ONNX Runtime, phát sinh chi phí \$0.00 USD.

### Phân tích 2 lỗi trên Dev, nguyên nhân và biện pháp khắc phục

1. **Lỗi 1 (Thất bại của Dense retrieval trên câu hỏi đa khái niệm — `q10`)**:
   - *Câu hỏi*: *"Does achieving transparency in an AI system automatically make it accurate, secure, or fair?"*
   - *Hiện tượng*: TF-IDF đạt Hit@1=1 và Hit@3=1, trong khi Dense retrieval trượt hoàn toàn (Hit@1=0, Hit@3=0).
   - *Nguyên nhân*: Embedding model `BAAI/bge-small-en-v1.5` phân bổ độ tương đồng cao cho các đoạn văn bản định nghĩa độc lập từng đặc tính "transparency", "accurate", "fair" (trang 20-22), thay vì đoạn văn bản tổng kết về sự đánh đổi (*trade-offs between characteristics*) ở trang 21. Do các từ khóa xuất hiện rải rác ở nhiều trang, biểu diễn vector mật bị loãng.
   - *Biện pháp & Đo lại*: Sử dụng **Hybrid RRF (Dense + BM25)**. Thuật toán BM25 gán trọng số cao cho sự đồng xuất hiện của tập từ khóa chuyên biệt (`transparency`, `accurate`, `secure`, `fair`, `trade-offs`), đưa chunk trang 21 trở lại vị trí số 1 trên bảng xếp hạng (Hit@1=1, Hit@3=1).

2. **Lỗi 2 (Sụp đổ Context Evidence Rate do giới hạn context budget — `q03`, `q05`, `q07`, `q14`)**:
   - *Hiện tượng*: Ở cả 4 câu hỏi này, cả TF-IDF và Dense retrieval đều tìm ra đúng chunk vàng ở Top-1 (Hit@1=1, Hit@3=1). Tuy nhiên, `Context Evidence Rate` đo được lại bằng **0%** (`context_ids = []`, hoàn toàn không có context gửi cho LLM)!
   - *Nguyên nhân*: Hàm `pack_context()` áp dụng trần cứng `budget = 800` ký tự. Trong khi đó, các semantic chunks từ Day 6 có kích thước đoạn văn khoảng 750–790 ký tự, khi cộng thêm phần tiêu đề metadata định tuyến (`[chunk_id | source | page]\n` tốn ~70 ký tự) thì tổng kích thước khối văn bản là ~850 ký tự ($> 800$). Do vượt trần, hàm `pack_context()` đã loại bỏ hoàn toàn chunk Top-1 ra khỏi ngữ cảnh, khiến Agent nhận context rỗng và buộc phải từ chối trả lời (false abstention).
   - *Giả thuyết & Đo lại*: Tăng `budget` từ 800 lên **1600 ký tự** (vừa vặn chứa 2 chunks đầy đủ). Kết quả trên Dev: `Context Evidence Rate` tăng vọt từ **53.3%** (8/15) lên **86.7%** (13/15); trên Test: tăng từ **60.0%** lên **80.0%**; đồng thời giảm false abstention của Agent từ 8 câu xuống chỉ còn 2 câu.

- **Lý do chốt cấu hình Hybrid RRF + Budget 1600**: Cấu hình này dung hòa tốt nhất giữa độ chính xác truy xuất (`Evidence Hit@3` đạt 93.3%), tỷ lệ cung cấp bằng chứng ngữ cảnh thực tế vào prompt cao nhất (86.7%), và chi phí tính toán hoàn toàn nằm trong ngưỡng chấp nhận được (thời gian truy xuất chỉ ~30ms, chạy local).

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - *Tool*: Agent được cung cấp công cụ `search` kết nối trực tiếp với retriever (`HybridRRFRetriever` hoặc `TfidfRetriever`), hỗ trợ tìm kiếm từ khóa và lọc metadata.
  - *Số lượt gọi & Điều kiện dừng*: Tối đa `max_searches = 4`. Agent dừng vòng lặp khi:
    1. LLM nhận diện context đã có đủ bằng chứng và phát sinh hành động `answer` kèm danh sách trích dẫn `citations`.
    2. LLM phát hiện câu hỏi thuộc nhóm thông tin không có trong tài liệu và phát sinh hành động `answer` rỗng citation (kích hoạt từ chối có nguyên tắc).
    3. Đạt ngưỡng giới hạn số lần gọi (`search_limit`) $\rightarrow$ tự động kích hoạt `status: "abstain"`.
  - *Cách chọn context*: Thu thập ứng viên từ các lượt tìm kiếm, loại trùng lặp theo `chunk_id`, đóng gói theo định dạng `[chunk_id | source | page]\n<text>` với ngân sách 1600 ký tự.
  - *Kiểm soát Grounding*: Hệ thống kiểm tra nghiêm ngặt: toàn bộ `citation` khai báo phải tồn tại trong tập context được nạp và phải xuất hiện tường minh dưới dạng `[chunk_id]` trong văn bản trả lời; nếu vi phạm, câu trả lời bị hủy và chuyển sang trạng thái từ chối.

### Bảng tổng hợp đánh giá Agent

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer 0/1/2 và TB (/2) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** (Dev) | 17/17 (100%) | 1.53 | 0: 8 \| 1: 1 \| 2: 6 (TB: 0.87) | 7/7 (100%) | 7/7 (100%) | 2/2 (100%) | 8/15 (53.3%) |
| **Cấu hình chọn** (Dev) | 17/17 (100%) | 1.88 | 0: 3 \| 1: 2 \| 2: 10 (TB: **1.47**) | 18/18 (100%) | 13/13 (100%) | 2/2 (100%) | 2/15 (13.3%) |
| **Baseline** (Test) | 7/7 (100%) | 1.57 | 0: 2 \| 1: 0 \| 2: 3 (TB: 1.20) | 3/3 (100%) | 3/3 (100%) | 2/2 (100%) | 2/5 (40.0%) |
| **Cấu hình chọn** (Test) | 7/7 (100%) | 1.71 | 0: 0 \| 1: 1 \| 2: 4 (TB: **1.80**) | 5/5 (100%) | 5/5 (100%) | 2/2 (100%) | **0/5 (0.0%)** |

---

### Case Studies chi tiết trên tập Test

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q16** | Có đáp án (trực tiếp) | `search("differences between AI system risks and traditional software risks Appendix B")` $\rightarrow$ 3 hits từ trang 43 | *"Risks from AI systems differ from traditional software risks in that AI systems introduce a set of risks that are not comprehensively addressed by current risk frameworks and approaches... [nist_ai_rmf_1_0_B_p43_c01]"* | **2 (Đạt)**: Trả lời chính xác, đầy đủ định nghĩa khác biệt từ Phụ lục B, trích dẫn chính xác `nist_ai_rmf_1_0_B_p43_c01`. |
| **q18** | Có đáp án (filter / điều kiện) | `search("MEASURE 1.1 prioritize risk measurement metrics")` $\rightarrow$ 3 hits từ trang 34 | *"According to subcategory MEASURE 1.1 of the NIST AI RMF 1.0, organizations should prioritize the implementation of risk measurement metrics by selecting approaches and metrics that are rigorous, repeatable, and tailored to the context of the AI system... [nist_ai_rmf_1_0_B_p34_c01]"* | **2 (Đạt)**: Bắt đúng tiểu mục MEASURE 1.1, nêu rõ tiêu chí ưu tiên trắc nghiệm và đo lường. |
| **q17** | Có đáp án (diễn đạt lại) | `search("MANAGE function core responsibility risk resources")` $\rightarrow$ 3 hits từ trang 37 | *"The core responsibility of the MANAGE function regarding risk resources involves prioritizing the treatment of documented AI risks based on their impact, likelihood, and the availability of resources... [nist_ai_rmf_1_0_B_p37_c01]"* | **1 (Đúng 1 phần)**: Trả lời đúng trọng tâm phân bổ nguồn lực nhưng trích dẫn trang 37 (thay vì trang 36 chứa định nghĩa tổng quan của bảng chức năng MANAGE). |
| **q_unans_test_01** | Thiếu bằng chứng (unanswerable) | `search("NIST mandatory commercial licensing procedure autonomous delivery drone")` $\rightarrow$ Trả về các chunk chung về độ an toàn | *"Không đủ bằng chứng để trả lời."* (Citations: `[]`) | **Đạt (Abstention đúng)**: Agent đã truy xuất kiểm chứng, nhận diện NIST không cấp phép thương mại cho drone và từ chối trả lời chính xác, không ảo giác thông tin. |
| **q_unans_test_02** | Thiếu bằng chứng (unanswerable) | `search("exact numerical score MMLU HumanEval benchmark MEASURE function")` $\rightarrow$ Trả về các chunk về định lượng đo lường rủi ro | *"Không đủ bằng chứng để trả lời."* (Citations: `[]`) | **Đạt (Abstention đúng)**: Tài liệu hướng dẫn phương pháp đo lường chung, không quy định điểm số cụ thể cho MMLU/HumanEval. Agent từ chối hợp lệ. |

---

## Chi phí, Tái lập và Đánh đổi

- **Đường dẫn artifacts, trace và tóm tắt kết quả:**
  - File dữ liệu chuẩn hóa: `data/chunks.json`, `data/questions.json`, `data/agent_cases.json`.
  - Kết quả thử nghiệm retrieval: `results/retrieval/summary.json` và các file trace chi tiết trong `results/retrieval/`.
  - Kết quả thử nghiệm agent:
    - Baseline Dev: `results/agent_dev_baseline/summary.json`
    - Cấu hình chọn Dev: `results/agent_dev_chosen/summary.json`
    - Baseline Test: `results/agent_test_baseline/summary.json`
    - Cấu hình chọn Test: `results/agent_test_chosen/summary.json`
    - Báo cáo tổng hợp: `results/agent_overall_summary.json`

- **Lệnh chạy tái lập toàn bộ bài thí nghiệm:**
  ```bash
  # 1. Chạy unit tests đảm bảo tính toàn vẹn hệ thống
  python -m unittest discover -s tests -v

  # 2. Chạy benchmark đối chiếu 5 cấu hình Retrieval
  python src/evaluate_retrieval.py

  # 3. Chạy đánh giá toàn diện Agent (Dev & Test)
  python src/evaluate_agent.py
  ```

- **Tổng kết tài nguyên & Chi phí API:**
  - *Chi phí tính toán Retrieval*: 0.00 USD (Embedding chạy local ONNX, Vector DB lưu file local).
  - *Chi phí LLM Controller*: Tổng cộng 48 lượt chạy agent (17 dev baseline + 17 dev chosen + 7 test baseline + 7 test chosen) tiêu tốn 73,573 prompt tokens và 6,144 completion tokens của mô hình `openai/gpt-4o-mini`, tổng chi phí thực tế là **0.01479 USD** (~370 VNĐ).
  - *Thời gian phản hồi*: Trung bình ~3.7s đến 5.5s cho một phiên hỏi đáp đa lượt của Agent.

- **Đánh đổi (Trade-offs) và Giới hạn:**
  - *Đánh đổi Context Budget*: Tăng budget từ 800 lên 1600 ký tự giúp triệt tiêu hiện tượng rớt bằng chứng do chunk dài, nâng tỷ lệ câu trả lời hoàn thiện từ 40% lên 80% trên test, đổi lại số prompt tokens nạp vào LLM tăng khoảng 45% (nhưng chi phí tăng không đáng kể do gpt-4o-mini rất rẻ).
  - *Đánh đổi Hybrid RRF vs Dense*: Thêm nhánh BM25 giúp khắc phục hoàn toàn điểm yếu của mô hình vector đối với các tên chuẩn/số hiệu (ISO 9000, ISO/IEC TS 5723), tuy nhiên độ trễ tìm kiếm tăng nhẹ từ 24ms lên ~32ms (hoàn toàn chấp nhận được).

---

## Nâng cấp Đột phá: VisRAG (Visual Document RAG), Prompt Engineering & Custom Skill

Nhằm vượt trội hơn các giải pháp RAG văn bản đơn thuần, dự án đã được nâng cấp toàn diện với các công nghệ tiên tiến nhất hiện nay:

### 1. Kiến trúc Đa phương thức VisRAG (Visual Document Retrieval-Augmented Generation)
* **Vấn đề của Text RAG truyền thống**: Khi trích xuất PDF thành văn bản thuần, các cấu trúc hình học, phân cấp đồng tâm (vòng tròn trong/ngoài trong Figure 2), mũi tên điều hướng giữa các tác nhân (Figure 3) và các bảng biểu đa cột đều bị xé nhỏ hoặc xáo trộn thứ tự.
* **Giải pháp VisRAG (`src/visrag.py`)**:
  - Tích hợp thư viện `pypdfium2` để render trực tiếp từng trang PDF thành hình ảnh chất lượng cao (150 DPI) với cơ chế bộ nhớ đệm tự động (`artifacts/page_cache/`).
  - Trang bị công cụ `inspect_visual_page(physical_page, visual_query)` cho Agent: Khi gặp các câu hỏi liên quan đến sơ đồ, hình vẽ (Figure 1, Figure 2, Figure 3) hoặc bảng biểu cấu trúc (Tables 1–4), Agent tự động "nhìn" trực tiếp trang PDF dưới dạng ảnh thông qua Vision Language Model (`gpt-4o-mini` Vision API).
  - Ngữ cảnh nạp vào LLM hỗ trợ song song hai loại bằng chứng: Text chunks (`[nist_ai_rmf_1_0_B_...]`) và Visual chunks (`[vis_page_XX]`).

### 2. Kỹ thuật Prompt Engineering Nâng cao & ReAct Loop (`src/visrag_agent.py`)
* **ReAct với Vùng suy nghĩ nội tại (`thought`)**: Trước khi đưa ra bất kỳ hành động nào, Agent bắt buộc phải suy luận tường minh xem câu hỏi cần tìm kiếm văn bản, quan sát trực quan hay đối chiếu chéo.
* **Định tuyến Trực quan Thông minh (Visual Routing)**: Hệ thống cung cấp chỉ mục điều hướng trang vật lý cho các sơ đồ trọng yếu trong prompt (Figure 1 tại trang 10; Figure 2 tại trang 15; Figure 3 tại trang 16; Tables 1–4 tại các trang 27, 31, 34, 37).
* **Kiểm định Nguồn kép (Dual-Source Grounding)**: Kiểm tra chốt chặn nghiêm ngặt: mọi khẳng định trong câu trả lời phải được bảo chứng bằng ít nhất một đoạn trích văn bản hoặc một bằng chứng trực quan thực tế từ trang PDF đã kiểm tra.

### 3. Đóng gói Kỹ năng Chuẩn hóa (Antigravity Custom Skill)
* Toàn bộ quy trình và mã nguồn VisRAG đã được đóng gói thành một Antigravity Skill độc lập tại:
  `skills/visrag-agent/SKILL.md`
* Hỗ trợ gọi trực tiếp trong IDE/CLI, kèm đầy đủ tài liệu API, sơ đồ luồng dữ liệu (Mermaid) và hướng dẫn chạy kiểm thử.

### 4. Kết quả Thực nghiệm Bộ Suite Trực quan VisRAG (`results/visrag/visrag_summary.json`)
* Đã thực nghiệm 5 ca kiểm thử trực quan phức tạp (Figure 1, Figure 2, Figure 3, Table Core Functions, và Unanswerable Circuit Diagram).
* **Kết quả tiêu biểu**:
  - Với câu hỏi về các vòng tròn đồng tâm trong Figure 2 (`vis_q03`), Agent đã quan sát trực quan trang 15, nhận diện chính xác vòng tròn trung tâm là *"People and Planet"*, vòng tròn thứ hai là *"AI Model"*, và vòng tròn ngoài cùng là các giai đoạn vòng đời (*"Plan & Design"*, *"Deploy & Use"*, v.v.), kèm trích dẫn chuẩn xác `[vis_page_15]`. Đây là thông tin mà text extraction hoàn toàn không thể tái hiện được mối quan hệ không gian.
  - Với câu hỏi bẫy về sơ đồ mạch GPU (`vis_q05`), Agent kiểm tra trang trực quan và từ chối hợp lệ 100% (*"Không đủ bằng chứng để trả lời."*).

### 5. Lệnh chạy thực nghiệm VisRAG:
```bash
# Chạy bộ benchmark đa phương thức VisRAG:
python src/evaluate_visrag.py
```


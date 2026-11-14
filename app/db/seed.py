from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.quiz import Category, Question, QuestionOption, Quiz, QuizQuestion

SEED_CATEGORIES = [
    {
        "name": "Nền tảng Python",
        "slug": "python-fundamentals",
        "description": "Python nâng cao cho AI Engineer: async/await, typing, generators, memory, vectorization.",
        "icon": "code",
        "color": "#3b82f6",
        "order_index": 1,
    },
    {
        "name": "Toán học cho AI",
        "slug": "math-for-ai",
        "description": "Đại số tuyến tính, giải tích ma trận, xác suất thống kê và tối ưu hóa cho Machine Learning.",
        "icon": "sigma",
        "color": "#8b5cf6",
        "order_index": 2,
    },
    {
        "name": "Học máy (Machine Learning)",
        "slug": "machine-learning",
        "description": "Supervised/Unsupervised Learning, Bias-Variance, Feature Engineering, XGBoost, Evaluation.",
        "icon": "git-branch",
        "color": "#10b981",
        "order_index": 3,
    },
    {
        "name": "Học sâu (Deep Learning)",
        "slug": "deep-learning",
        "description": "PyTorch, Backpropagation, Normalization, Optimizers, Attention & Transformer internals.",
        "icon": "cpu",
        "color": "#f59e0b",
        "order_index": 4,
    },
    {
        "name": "Xử lý ngôn ngữ tự nhiên & LLM",
        "slug": "nlp-llm",
        "description": "Tokenization, Embeddings, RAG, LoRA/QLoRA fine-tuning, KV Cache và Decoding strategies.",
        "icon": "message-square",
        "color": "#ec4899",
        "order_index": 5,
    },
    {
        "name": "Thị giác máy tính (Computer Vision)",
        "slug": "computer-vision",
        "description": "CNN architectures, ViT, Object Detection, Segmentation, Multimodal Vision-Language Models.",
        "icon": "eye",
        "color": "#06b6d4",
        "order_index": 6,
    },
    {
        "name": "MLOps & Triển khai mô hình",
        "slug": "mlops",
        "description": "Model serving, vLLM/ONNX, Docker, CI/CD, Drift monitoring, Feature Stores & Evaluation pipelines.",
        "icon": "server",
        "color": "#64748b",
        "order_index": 7,
    },
    {
        "name": "AI Agents & Trợ lý thông minh",
        "slug": "ai-agents",
        "description": "Tool calling, ReAct, Multi-agent orchestration, MCP, Memory, Guardrails & Agent evaluation.",
        "icon": "bot",
        "color": "#6366f1",
        "order_index": 8,
    },
]


SEED_QUESTIONS: dict[str, list[dict]] = {
    "python-fundamentals": [
        {
            "content": "Kết quả của đoạn code Python sau khi dùng default mutable argument là gì?",
            "code_snippet": "def add_item(item, target=[]):\n    target.append(item)\n    return target\n\nprint(add_item('a'))\nprint(add_item('b'))",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "Default arguments trong Python được evaluate một lần duy nhất tại thời điểm định nghĩa hàm (`def`), nên list `target` được chia sẻ qua các lần gọi. Best practice là dùng `target: list | None = None`.",
            "options": [
                ("['a'] và ['b']", False),
                ("['a'] và ['a', 'b']", True),
                ("Lỗi TypeError ở lần gọi thứ hai", False),
                ("['a'] và []", False),
            ],
        },
        {
            "content": "Trong Python `asyncio`, điều gì xảy ra nếu bạn gọi `time.sleep(5)` trực tiếp bên trong một `async def` coroutine đang chạy trên event loop?",
            "code_snippet": "async def fetch_data():\n    time.sleep(5)\n    return 'done'",
            "code_language": "python",
            "difficulty": "medium",
            "explanation": "`time.sleep()` là lời gọi đồng bộ (blocking). Nó chặn toàn bộ thread của event loop, khiến mọi coroutine khác không thể chạy song song. Hãy dùng `await asyncio.sleep(5)` hoặc `await asyncio.to_thread(...)`.",
            "options": [
                ("Event loop tự động chuyển sang coroutine khác trong lúc sleep", False),
                ("Toàn bộ event loop bị block trong 5 giây, các task khác phải chờ", True),
                ("Python tự động wrap `time.sleep` thành `asyncio.sleep`", False),
                ("Ném ra RuntimeError: blocking call in async function", False),
            ],
        },
        {
            "content": "Ưu điểm chính của Generator (`yield`) khi xử lý tập dữ liệu lớn (ví dụ 10GB JSONL corpus) so với List comprehension là gì?",
            "code_snippet": "def stream_lines(path: str):\n    with open(path, 'r', encoding='utf-8') as f:\n        for line in f:\n            yield line.strip()",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "Generator thực thi theo cơ chế lazy evaluation — chỉ giữ một phần tử tại một thời điểm trong bộ nhớ (O(1) memory), thay vì load toàn bộ danh sách vào RAM như list comprehension.",
            "options": [
                ("Chạy đa luồng tự động qua mọi CPU core", False),
                ("Tiết kiệm bộ nhớ O(1) nhờ lazy evaluation từng phần tử", True),
                ("Cho phép index ngẫu nhiên `gen[1000]` nhanh hơn list", False),
                ("Tự động lưu cache kết quả vào đĩa", False),
            ],
        },
        {
            "content": "Vì sao Python GIL (Global Interpreter Lock) ít ảnh hưởng tới tốc độ huấn luyện PyTorch hoặc gọi LLM API qua `httpx` async?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "GIL chỉ khóa bytecode Python trong một process. Các thư viện C/CUDA như NumPy, PyTorch giải phóng GIL khi tính toán tensor trên C++/GPU, còn `asyncio`/`httpx` giải phóng luồng khi chờ I/O mạng.",
            "options": [
                ("Vì Python 3 tự động tắt GIL cho mọi hàm có type hints", False),
                ("Vì tác vụ nặng chạy trong C/CUDA (nhả GIL) và I/O mạng dùng async non-blocking", True),
                ("Vì PyTorch tự động biên dịch Python sang JavaScript", False),
                ("Vì GIL chỉ áp dụng cho biến toàn cục kiểu string", False),
            ],
        },
        {
            "content": "Đoạn code sử dụng `functools.lru_cache` dưới đây có vấn đề gì nếu truyền một `list` làm tham số?",
            "code_snippet": "from functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef compute_score(tokens: list[str]) -> int:\n    return len(tokens)",
            "code_language": "python",
            "difficulty": "medium",
            "explanation": "`lru_cache` dùng dictionary để cache theo arguments nên mọi argument phải hashable. `list` là mutable/unhashable nên sẽ ném `TypeError: unhashable type: 'list'`. Nên truyền `tuple[str, ...]`.",
            "options": [
                ("Trả về 0 thay vì độ dài list", False),
                ("Ném TypeError: unhashable type: 'list'", True),
                ("Tự động ép kiểu list thành set", False),
                ("Gây memory leak ngay lập tức", False),
            ],
        },
        {
            "content": "Trong Python typing, `Protocol` (PEP 544) khác với Abstract Base Class (`ABC`) ở điểm nào quan trọng nhất?",
            "code_snippet": "from typing import Protocol\n\nclass Embedder(Protocol):\n    def embed(self, text: str) -> list[float]: ...",
            "code_language": "python",
            "difficulty": "hard",
            "explanation": "`Protocol` hỗ trợ structural subtyping (duck typing tĩnh): bất kỳ class nào có method `embed(self, text: str) -> list[float]` đều thỏa mãn `Embedder` mà không cần kế thừa trực tiếp từ `Embedder`.",
            "options": [
                ("Protocol chỉ hoạt động ở runtime, không dùng được với mypy/pyright", False),
                ("Protocol dùng structural subtyping: không cần kế thừa tường minh", True),
                ("Protocol bắt buộc class con phải dùng `@dataclass`", False),
                ("Protocol nhanh hơn ABC 10 lần khi khởi tạo instance", False),
            ],
        },
    ],
    "math-for-ai": [
        {
            "content": "Trong phép nhân ma trận `A @ B` với `A` kích thước `(m, n)` và `B` kích thước `(n, p)`, kích thước của ma trận kết quả là gì?",
            "code_snippet": "import numpy as np\nA = np.zeros((32, 768))\nB = np.zeros((768, 10))\nC = A @ B",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "Nhân ma trận `(m, n) @ (n, p)` yêu cầu chiều trong bằng nhau (`n == n`) và cho kết quả kích thước `(m, p)` — ở đây là `(32, 10)`.",
            "options": [
                ("(768, 768)", False),
                ("(32, 10)", True),
                ("(10, 32)", False),
                ("(32, 768)", False),
            ],
        },
        {
            "content": "Cosine Similarity giữa hai embedding vector `u` và `v` đo lường đại lượng nào?",
            "code_snippet": "cos_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "Cosine similarity tính cosin của góc giữa hai vector trong khoảng `[-1, 1]`, bất biến với độ dài (magnitude) vector, rất phù hợp để so sánh hướng ngữ nghĩa trong embedding space.",
            "options": [
                ("Khoảng cách Euclid tuyệt đối giữa hai điểm", False),
                ("Cosin của góc giữa hai vector, đo sự tương đồng về hướng", True),
                ("Tích có hướng (cross product) trong không gian 3D", False),
                ("Định thức của ma trận tạo bởi u và v", False),
            ],
        },
        {
            "content": "Tại sao trong Scaled Dot-Product Attention lại chia `Q @ K^T` cho `sqrt(d_k)` trước khi đưa qua Softmax?",
            "code_snippet": "scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)\nweights = F.softmax(scores, dim=-1)",
            "code_language": "python",
            "difficulty": "medium",
            "explanation": "Khi `d_k` lớn, phương sai của dot-product `q·k` tăng tuyến tính theo `d_k`, đẩy các giá trị vào vùng bão hòa của Softmax khiến gradient cực nhỏ (vanishing gradient). Chia cho `sqrt(d_k)` đưa phương sai về 1.",
            "options": [
                ("Để giảm độ phức tạp tính toán từ O(n^2) xuống O(n)", False),
                ("Để giữ phương sai của dot-product ổn định, tránh bão hòa Softmax và vanishing gradient", True),
                ("Để đảm bảo ma trận attention luôn đối xứng", False),
                ("Để chuẩn hóa vector Q và K thành vector đơn vị", False),
            ],
        },
        {
            "content": "Ý nghĩa của Eigenvalues (trị riêng) trong PCA (Principal Component Analysis) khi phân rã ma trận hiệp phương sai là gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Mỗi eigenvector tương ứng với một trục chính (principal component), còn eigenvalue tương ứng biểu thị lượng phương sai (variance) của dữ liệu được giữ lại dọc theo trục đó.",
            "options": [
                ("Số lượng chiều tối thiểu của dữ liệu gốc", False),
                ("Lượng phương sai của dữ liệu dọc theo eigenvector tương ứng", True),
                ("Hệ số regularization L2 tối ưu", False),
                ("Learning rate tối ưu cho mỗi feature", False),
            ],
        },
        {
            "content": "KL Divergence `D_KL(P || Q)` có tính chất nào sau đây?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "hard",
            "explanation": "KL Divergence luôn không âm (`>= 0`, bằng 0 khi `P == Q`), nhưng KHÔNG đối xứng (`D_KL(P || Q) != D_KL(Q || P)`), nên nó không phải là một distance metric thực thụ.",
            "options": [
                ("Đối xứng và luôn nằm trong đoạn [0, 1]", False),
                ("Không âm (>= 0) nhưng không đối xứng: D_KL(P||Q) != D_KL(Q||P)", True),
                ("Có thể nhận giá trị âm khi Q có entropy lớn hơn P", False),
                ("Bằng với khoảng cách Wasserstein-1", False),
            ],
        },
        {
            "content": "Trong tối ưu hóa hàm lồi (convex optimization), điều kiện cần bậc nhất để `x*` là điểm cực tiểu toàn cục của hàm khả vi `f(x)` không ràng buộc là gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Với hàm lồi khả vi trên toàn miền xác định, điểm dừng nơi gradient bằng vector 0 (`∇f(x*) = 0`) chính là cực tiểu toàn cục (global minimum).",
            "options": [
                ("Hessian matrix có định thức bằng 0", False),
                ("Gradient tại x* bằng 0: ∇f(x*) = 0", True),
                ("Giá trị f(x*) = 0", False),
                ("Chuẩn L2 của x* bằng 1", False),
            ],
        },
    ],
    "machine-learning": [
        {
            "content": "Hiện tượng Data Leakage xảy ra trong đoạn code preprocessing scikit-learn nào sau đây?",
            "code_snippet": "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "`scaler.fit_transform(X)` tính mean và std trên toàn bộ dataset gồm cả tập test trước khi `train_test_split`. Đúng chuẩn là `split` trước, `fit_transform` trên `X_train` và chỉ `transform` trên `X_test` (hoặc dùng `Pipeline`).",
            "options": [
                ("Dùng `test_size=0.2` là quá nhỏ", False),
                ("Fit `StandardScaler` trên toàn bộ `X` trước khi chia train/test làm rò rỉ thống kê tập test", True),
                ("`StandardScaler` không hỗ trợ mảng 2D", False),
                ("Thiếu tham số `stratify` trong `train_test_split`", False),
            ],
        },
        {
            "content": "Khi bài toán phân loại nhị phân có mất cân bằng lớp nghiêm trọng (ví dụ 99% negative, 1% positive fraud), metric nào KHÔNG phù hợp làm chỉ số đánh giá chính?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "Với 99% negative, một mô hình ngây thơ luôn dự đoán negative sẽ đạt Accuracy 99% nhưng hoàn toàn vô dụng. Cần dùng PR-AUC, F1-score, Precision/Recall hoặc ROC-AUC.",
            "options": [
                ("PR-AUC (Precision-Recall AUC)", False),
                ("Accuracy", True),
                ("F1-Score của lớp thiểu số", False),
                ("Recall @ fixed Precision", False),
            ],
        },
        {
            "content": "Sự khác biệt cốt lõi giữa Bagging (như Random Forest) và Boosting (như XGBoost / LightGBM) là gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Bagging huấn luyện nhiều cây độc lập song song trên các bootstrap sample để giảm variance. Boosting huấn luyện các cây tuần tự, mỗi cây mới sửa sai số (residual/gradient) của các cây trước để giảm bias.",
            "options": [
                ("Bagging huấn luyện tuần tự giảm bias; Boosting song song giảm variance", False),
                ("Bagging huấn luyện song song giảm variance; Boosting huấn luyện tuần tự giảm bias", True),
                ("Bagging chỉ dùng cho regression; Boosting chỉ dùng cho classification", False),
                ("Random Forest dùng neural network bên dưới", False),
            ],
        },
        {
            "content": "L1 Regularization (Lasso) thường tạo ra trọng số thưa (sparse weights — nhiều hệ số bằng đúng 0) hơn L2 Regularization (Ridge) vì sao?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Hình học vùng ràng buộc của chuẩn L1 là hình thoi (diamond) có các đỉnh nhọn nằm trên các trục tọa độ, nên đường đồng mức hàm loss dễ tiếp xúc tại các đỉnh nơi một số trọng số bằng 0.",
            "options": [
                ("Vì L1 có learning rate lớn hơn L2", False),
                ("Vì vùng ràng buộc L1 có các góc nhọn nằm trên trục tọa độ (subgradient hằng số tới 0)", True),
                ("Vì L1 chỉ áp dụng được cho cây quyết định", False),
                ("Vì L2 không khả vi tại mọi điểm", False),
            ],
        },
        {
            "content": "Khi mô hình có Training Loss thấp (1%) nhưng Validation Loss cao (25%), hành động nào giúp khắc phục hiệu quả nhất?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "Training tốt nhưng Validation kém là dấu hiệu điển hình của Overfitting (High Variance). Giải pháp gồm tăng dữ liệu train, tăng regularization (dropout/weight decay), giảm độ phức tạp mô hình, hoặc early stopping.",
            "options": [
                ("Tăng số tầng/tham số của mô hình để fit kỹ hơn", False),
                ("Giảm regularization và train thêm nhiều epoch", False),
                ("Áp dụng regularization, thêm dữ liệu, data augmentation hoặc early stopping", True),
                ("Bỏ tập validation để tập trung vào training loss", False),
            ],
        },
        {
            "content": "SHAP values trong Explainable AI (XAI) được xây dựng dựa trên khái niệm lý thuyết nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "hard",
            "explanation": "SHAP (SHapley Additive exPlanations) dựa trên Shapley values từ Lý thuyết trò chơi hợp tác (Cooperative Game Theory), phân bổ công bằng đóng góp biên của từng feature vào dự đoán.",
            "options": [
                ("Principal Component Analysis", False),
                ("Shapley values trong Cooperative Game Theory", True),
                ("Expectation-Maximization", False),
                ("Markov Chain Monte Carlo", False),
            ],
        },
    ],
    "deep-learning": [
        {
            "content": "Trong PyTorch, vì sao phải gọi `optimizer.zero_grad()` ở mỗi bước training loop trước `loss.backward()`?",
            "code_snippet": "for x, y in loader:\n    optimizer.zero_grad()\n    loss = criterion(model(x), y)\n    loss.backward()\n    optimizer.step()",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "PyTorch mặc định CỘNG DỒN (accumulate) gradient vào `.grad` mỗi lần gọi `.backward()` (hữu ích cho gradient accumulation). Nếu không gọi `zero_grad()`, gradient các batch cũ sẽ cộng dồn làm sai hướng cập nhật.",
            "options": [
                ("Để giải phóng toàn bộ VRAM của GPU", False),
                ("Vì PyTorch cộng dồn gradient vào `.grad` qua các lần `backward()` nếu không xóa", True),
                ("Để chuyển model sang chế độ `eval()`", False),
                ("Để reset trọng số của model về 0", False),
            ],
        },
        {
            "content": "Khi suy luận (inference) với mô hình PyTorch chứa `Dropout` và `BatchNorm`, cần làm gì để đảm bảo kết quả đúng và tiết kiệm bộ nhớ?",
            "code_snippet": "model.eval()\nwith torch.inference_mode():\n    preds = model(x)",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "`model.eval()` tắt Dropout và chuyển BatchNorm sang dùng running mean/var; `torch.inference_mode()` (hoặc `no_grad()`) tắt autograd graph để giảm RAM/VRAM và tăng tốc.",
            "options": [
                ("Chỉ cần gọi `model.train(True)`", False),
                ("Gọi `model.eval()` và bọc trong `torch.inference_mode()` (hoặc `torch.no_grad()`)", True),
                ("Xóa tất cả lớp Dropout khỏi file code", False),
                ("Chạy `loss.backward()` sau mỗi lần dự đoán", False),
            ],
        },
        {
            "content": "Sự khác biệt chính giữa BatchNorm và LayerNorm trong kiến trúc Transformer là gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "BatchNorm chuẩn hóa theo chiều batch (phụ thuộc batch size và gặp khó khăn với chuỗi độ dài thay đổi), còn LayerNorm chuẩn hóa qua các feature/embedding của từng token độc lập với batch size — rất lý tưởng cho Transformer.",
            "options": [
                ("LayerNorm chuẩn hóa qua các feature của từng sample, độc lập với batch size", True),
                ("BatchNorm không có tham số học được (gamma, beta)", False),
                ("LayerNorm chỉ dùng được cho ảnh RGB", False),
                ("LayerNorm yêu cầu batch size tối thiểu là 64", False),
            ],
        },
        {
            "content": "AdamW cải tiến điểm gì quan trọng so với Adam truyền thống?",
            "code_snippet": "optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)",
            "code_language": "python",
            "difficulty": "medium",
            "explanation": "Adam truyền thống cộng L2 penalty trực tiếp vào gradient, khiến hệ số weight decay bị chia tỉ lệ lại bởi adaptive learning rate. AdamW tách rời (decouple) weight decay ra khỏi bước cập nhật gradient thích nghi.",
            "options": [
                ("Loại bỏ hoàn toàn momentum bậc 1", False),
                ("Tách rời (decoupled) Weight Decay ra khỏi adaptive gradient update", True),
                ("Chỉ lưu gradient dưới dạng int8", False),
                ("Không cần learning rate scheduler", False),
            ],
        },
        {
            "content": "Mixed Precision Training (`bf16` vs `fp16`) vì sao `bfloat16` thường ổn định hơn `fp16` khi train LLM mà không cần Loss Scaling?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "hard",
            "explanation": "`bfloat16` giữ nguyên 8 bit exponent như `fp32` nên có dải biểu diễn động (dynamic range) tương đương `fp32`, tránh hiện tượng gradient underflow/overflow thường gặp ở `fp16` (chỉ có 5 bit exponent).",
            "options": [
                ("Vì bf16 có nhiều bit mantissa hơn fp32", False),
                ("Vì bf16 giữ nguyên 8-bit exponent như fp32 nên có dynamic range rộng", True),
                ("Vì bf16 chạy trên CPU nhanh hơn GPU", False),
                ("Vì bf16 tự động lượng tử hóa activation về 4-bit", False),
            ],
        },
        {
            "content": "Residual Connection (`y = x + F(x)`) trong ResNet và Transformer giải quyết vấn đề cốt lõi nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "Đường tắt (skip connection) tạo luồng truyền gradient trực tiếp (`∂y/∂x = 1 + ∂F/∂x`) xuyên qua hàng chục/trăm tầng, giúp giảm mạnh hiện tượng vanishing gradient và thoái hóa khi mạng rất sâu.",
            "options": [
                ("Giảm số lượng tham số của lớp Linear đi một nửa", False),
                ("Tạo đường truyền gradient trực tiếp, giảm vanishing gradient ở mạng sâu", True),
                ("Thay thế hoàn toàn hàm kích hoạt phi tuyến", False),
                ("Tự động tăng kích thước ảnh đầu vào", False),
            ],
        },
    ],
    "nlp-llm": [
        {
            "content": "Trong kiến trúc Decoder-only LLM (như Llama, GPT), **KV Cache** giúp tối ưu giai đoạn nào và đánh đổi tài nguyên nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "KV Cache lưu lại Key và Value của các token đã xử lý ở các bước decode trước, giúp bước sinh token mới không cần tính lại toàn bộ prefix (giảm từ O(t^2) xuống O(t) mỗi bước), đánh đổi bằng bộ nhớ GPU VRAM.",
            "options": [
                ("Tăng tốc giai đoạn Training, tốn thêm CPU", False),
                ("Tăng tốc giai đoạn Decode/Inference bằng cách tránh tính lại K, V cũ, đánh đổi bằng VRAM", True),
                ("Giảm kích thước vocabulary của tokenizer", False),
                ("Loại bỏ hoàn toàn lớp Softmax ở đầu ra", False),
            ],
        },
        {
            "content": "LoRA (Low-Rank Adaptation) đóng băng trọng số gốc `W_0` và học thêm hai ma trận `B @ A` hạng thấp `r << d`. Lợi ích lớn nhất là gì?",
            "code_snippet": "# h = W_0 * x + (alpha / r) * (B @ A) * x\n# W_0: (d, k), B: (d, r), A: (r, k)",
            "code_language": "python",
            "difficulty": "medium",
            "explanation": "Số tham số cần train giảm hàng chục đến hàng trăm lần (chỉ lưu optimizer states cho `A` và `B`), giúp fine-tune LLM lớn trên 1 GPU, và khi suy luận có thể merge `W = W_0 + (alpha/r)*B@A` với zero latency overhead.",
            "options": [
                ("Tăng kích thước checkpoint cần lưu lên gấp 3 lần", False),
                ("Giảm mạnh VRAM & số tham số trainable, có thể merge vào W_0 khi inference mà không tăng latency", True),
                ("Chỉ dùng được cho tác vụ dịch máy", False),
                ("Bắt buộc phải retrain lại tokenizer từ đầu", False),
            ],
        },
        {
            "content": "Trong hệ thống RAG (Retrieval-Augmented Generation), kỹ thuật **Hybrid Search** thường kết hợp hai phương pháp nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "Hybrid Search kết hợp Dense Vector Search (tìm kiếm ngữ nghĩa bằng embedding) và Sparse Keyword Search (như BM25 cho tên riêng, mã lỗi, thuật ngữ chính xác), thường trộn kết quả bằng Reciprocal Rank Fusion (RRF).",
            "options": [
                ("CNN và RNN", False),
                ("Dense Vector Search (semantic) + Sparse Search như BM25 (lexical)", True),
                ("K-Means và DBSCAN", False),
                ("Greedy decoding và Beam search", False),
            ],
        },
        {
            "content": "Tham số `temperature` trong LLM decoding ảnh hưởng như thế nào đến phân phối xác suất đầu ra?",
            "code_snippet": "probs = softmax(logits / temperature)",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "`temperature -> 0` làm phân phối nhọn hơn (deterministic, ưu tiên token xác suất cao nhất, hợp với code/JSON). `temperature > 1` làm phân phối phẳng hơn, tăng tính đa dạng/ngẫu nhiên.",
            "options": [
                ("Temperature càng cao, mô hình càng lặp lại và tất định", False),
                ("Temperature thấp (<0.3) tăng tính tất định/chính xác; cao (>1.0) tăng độ ngẫu nhiên", True),
                ("Temperature kiểm soát số token tối đa trong context window", False),
                ("Temperature chỉ áp dụng trong lúc pre-training", False),
            ],
        },
        {
            "content": "Byte-Pair Encoding (BPE) giải quyết bài toán gì trong NLP hiện đại?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "BPE chia từ thành các đơn vị con (subword) dựa trên tần suất cặp ký tự/byte xuất hiện nhiều nhất, cân bằng kích thước từ vựng và xử lý được từ hiếm/từ mới (Out-of-Vocabulary) mà không bị `[UNK]`.",
            "options": [
                ("Mã hóa âm thanh thành spectrogram", False),
                ("Subword tokenization giúp xử lý từ hiếm/mới (tránh OOV) với kích thước vocab gọn", True),
                ("Nén trọng số mô hình từ fp32 xuống fp4", False),
                ("Sắp xếp thứ tự các tài liệu trong vector database", False),
            ],
        },
        {
            "content": "Trong RAG, vì sao bước **Cross-Encoder Reranker** thường cho độ chính xác cao hơn Bi-Encoder Embedding Search nhưng chỉ chạy trên top-k (ví dụ top 50)?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "hard",
            "explanation": "Cross-Encoder ghép `(query, doc)` đi qua full self-attention nên tương tác sâu từng token (rất chính xác) nhưng không thể pre-compute embedding trước -> chậm O(N), nên chỉ dùng để rerank top-k ứng viên từ Bi-Encoder.",
            "options": [
                ("Vì Cross-Encoder nhỏ hơn Bi-Encoder 100 lần", False),
                ("Vì Cross-Encoder tính full attention trực tiếp giữa query và doc nên chính xác hơn nhưng không pre-compute được", True),
                ("Vì Bi-Encoder không dùng Transformer", False),
                ("Vì Cross-Encoder chỉ hỗ trợ tiếng Anh", False),
            ],
        },
    ],
    "computer-vision": [
        {
            "content": "Nếu áp dụng lớp `Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)` lên tensor ảnh `(1, 3, 224, 224)`, kích thước đầu ra là gì?",
            "code_snippet": "conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)\nout = conv(torch.randn(1, 3, 224, 224))",
            "code_language": "python",
            "difficulty": "easy",
            "explanation": "Công thức: `H_out = floor((H + 2*P - K)/S) + 1 = (224 + 2 - 3)/1 + 1 = 224`. Channel đầu ra bằng `out_channels=64`, nên shape là `(1, 64, 224, 224)`.",
            "options": [
                ("(1, 64, 222, 222)", False),
                ("(1, 64, 224, 224)", True),
                ("(1, 3, 224, 224)", False),
                ("(1, 64, 112, 112)", False),
            ],
        },
        {
            "content": "Vision Transformer (ViT) chuyển đổi ảnh 2D thành chuỗi token cho Transformer Encoder bằng cách nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "ViT chia ảnh thành lưới các patch vuông không chồng lấn (ví dụ 16x16), flatten và chiếu tuyến tính (linear projection) mỗi patch thành một patch embedding, cộng thêm positional embedding.",
            "options": [
                ("Chuyển từng pixel thành một token riêng biệt", False),
                ("Chia ảnh thành các patch cố định (vd 16x16), linear project thành patch embeddings + positional embeddings", True),
                ("Chạy thuật toán SIFT để trích xuất keypoints", False),
                ("Dùng RNN quét từng hàng pixel", False),
            ],
        },
        {
            "content": "Trong Object Detection, thuật toán **NMS (Non-Maximum Suppression)** thực hiện nhiệm vụ gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "NMS loại bỏ các bounding box trùng lặp dự đoán cùng một đối tượng: giữ box có confidence score cao nhất và loại bỏ các box cùng lớp có IoU (Intersection over Union) vượt ngưỡng threshold.",
            "options": [
                ("Tăng độ phân giải ảnh đầu vào trước khi detect", False),
                ("Loại bỏ các bounding box chồng lấn dư thừa dựa trên confidence score và ngưỡng IoU", True),
                ("Chuyển đổi ảnh màu sang ảnh xám", False),
                ("Tính hàm loss cho lớp phân loại", False),
            ],
        },
        {
            "content": "Sự khác biệt giữa Semantic Segmentation và Instance Segmentation là gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "Semantic Segmentation gán nhãn lớp cho từng pixel (mọi chiếc xe đều cùng một nhãn `car`). Instance Segmentation phân biệt từng cá thể riêng lẻ (`car_1`, `car_2`).",
            "options": [
                ("Semantic phân biệt từng cá thể riêng lẻ; Instance thì không", False),
                ("Instance Segmentation tách riêng từng cá thể cùng lớp, còn Semantic Segmentation chỉ phân loại theo lớp pixel", True),
                ("Semantic chỉ vẽ bounding box hình chữ nhật", False),
                ("Instance Segmentation không dự đoán mask pixel", False),
            ],
        },
        {
            "content": "Mô hình **CLIP** của OpenAI học biểu diễn đa phương thức (Image-Text) bằng hàm loss nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "CLIP sử dụng Contrastive Loss (InfoNCE đối xứng) trên ma trận cosine similarity giữa N cặp ảnh-chú thích trong một batch: tối đa hóa similarity của N cặp đúng trên đường chéo và tối thiểu hóa N^2 - N cặp sai.",
            "options": [
                ("Mean Squared Error giữa pixel và ký tự", False),
                ("Symmetric Contrastive Loss (InfoNCE) đẩy cặp ảnh-văn bản khớp nhau lại gần", True),
                ("CTC Loss (Connectionist Temporal Classification)", False),
                ("Triplet Margin Loss với 1 mẫu âm cố định", False),
            ],
        },
        {
            "content": "Depthwise Separable Convolution (dùng trong MobileNet) giảm chi phí tính toán FLOPs bằng cách nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "hard",
            "explanation": "Tách phép tích chập chuẩn thành 2 bước: Depthwise Conv (lọc không gian trên từng channel độc lập) + Pointwise Conv 1x1 (trộn thông tin giữa các channel), giảm FLOPs khoảng 8-9 lần với kernel 3x3.",
            "options": [
                ("Bỏ qua hoàn toàn hàm ReLU", False),
                ("Tách thành Depthwise Conv theo từng kênh và Pointwise 1x1 Conv để trộn kênh", True),
                ("Chỉ dùng kernel kích thước 1x1 cho mọi tầng", False),
                ("Giảm bit-width của ảnh xuống nhị phân", False),
            ],
        },
    ],
    "mlops": [
        {
            "content": "Trong Docker cho dịch vụ AI/FastAPI, vì sao nên dùng **multi-stage build** và user không phải `root`?",
            "code_snippet": "FROM python:3.12-slim AS builder\n...\nFROM python:3.12-slim\nRUN useradd -m appuser\nUSER appuser",
            "code_language": "dockerfile",
            "difficulty": "easy",
            "explanation": "Multi-stage build loại bỏ compiler/build cache khỏi image cuối (image nhẹ, ít CVE). Chạy bằng non-root user (`appuser`) tuân thủ nguyên tắc least privilege nếu container bị khai thác.",
            "options": [
                ("Để container tự động có quyền truy cập GPU host mà không cần nvidia-container-toolkit", False),
                ("Giảm kích thước image, giảm bề mặt tấn công CVE và hạn chế đặc quyền (least privilege)", True),
                ("Vì Docker không cho phép `pip install` trên image single-stage", False),
                ("Để chạy FastAPI bằng quyền kernel root bên trong container", False),
            ],
        },
        {
            "content": "Sự khác biệt giữa **Data Drift** và **Concept Drift** trong giám sát mô hình production là gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Data Drift là sự thay đổi phân phối đầu vào `P(X)` (ví dụ user dùng từ khóa mới). Concept Drift là sự thay đổi mối quan hệ `P(Y|X)` giữa input và nhãn mục tiêu (ví dụ hành vi gian lận thay đổi cách thức).",
            "options": [
                ("Data Drift là thay đổi P(X) đầu vào; Concept Drift là thay đổi mối quan hệ P(Y|X)", True),
                ("Data Drift chỉ xảy ra khi mất database; Concept Drift là lỗi code", False),
                ("Cả hai đều có nghĩa giống hệt nhau là model bị hết RAM", False),
                ("Concept Drift là khi tăng batch size", False),
            ],
        },
        {
            "content": "Tại sao **Continuous Batching** (như trong vLLM / TGI) giúp tăng throughput serving LLM gấp nhiều lần so với Static Batching?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Với Static Batching, mọi request trong batch phải chờ request sinh dài nhất hoàn thành rồi mới trả kết quả/nhận batch mới. Continuous Batching chèn request mới vào slot ngay khi một request trong batch kết thúc ở cấp độ iteration.",
            "options": [
                ("Vì nó bỏ qua bước tokenization", False),
                ("Vì request mới được nạp ngay vào batch ở từng bước decode khi có request vừa hoàn thành", True),
                ("Vì nó ép mọi câu trả lời có cùng độ dài 128 token", False),
                ("Vì nó chuyển toàn bộ model sang chạy trên CPU", False),
            ],
        },
        {
            "content": "Lợi ích chính của **Feature Store** (như Feast, Tecton) là giải quyết vấn đề nào giữa Offline Training và Online Serving?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Feature Store đảm bảo định nghĩa biến đổi feature nhất quán giữa môi trường train (offline) và inference (online), loại bỏ **Training-Serving Skew** và rò rỉ dữ liệu theo thời gian (point-in-time joins).",
            "options": [
                ("Thay thế hoàn toàn GitHub để lưu source code", False),
                ("Loại bỏ Training-Serving Skew nhờ đồng bộ logic feature offline & online", True),
                ("Tự động gán nhãn dữ liệu không cần con người", False),
                ("Nén trọng số mô hình PyTorch sang ONNX", False),
            ],
        },
        {
            "content": "Khi triển khai Canary Deployment cho phiên bản mô hình AI mới (`v2`), luồng traffic được điều phối như thế nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "Canary Deployment chuyển một tỷ lệ nhỏ traffic thực tế (ví dụ 5-10%) sang `v2` để theo dõi latency, error rate và chất lượng dự đoán trước khi rollout 100%, giúp giảm thiểu rủi ro.",
            "options": [
                ("Tắt toàn bộ `v1` trước rồi mới bật `v2`", False),
                ("Chuyển một phần nhỏ traffic (vd 5%) sang `v2` để kiểm chứng trước khi rollout toàn bộ", True),
                ("Gửi mọi request tới `v2` nhưng bỏ qua kết quả", False),
                ("Chạy `v2` chỉ trên môi trường localhost của developer", False),
            ],
        },
        {
            "content": "PagedAttention trong vLLM giải quyết nút thắt cổ chai nào của bộ nhớ GPU khi phục vụ nhiều người dùng đồng thời?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "hard",
            "explanation": "PagedAttention chia KV Cache thành các block không cần liên tục nhau trong VRAM (tương tự virtual memory paging của OS), loại bỏ phân mảnh bộ nhớ trong/ngoài và cho phép chia sẻ prefix KV blocks.",
            "options": [
                ("Giảm kích thước file trọng số trên ổ cứng SSD", False),
                ("Giảm phân mảnh VRAM của KV Cache bằng cách cấp phát theo block/page không liên tục", True),
                ("Tăng băng thông mạng giữa client và server", False),
                ("Tự động fine-tune mô hình theo từng user", False),
            ],
        },
    ],
    "ai-agents": [
        {
            "content": "Mô hình **ReAct** (Reasoning + Acting) trong AI Agent hoạt động theo chu trình lặp nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "ReAct xen kẽ bước suy luận (`Thought`) với bước gọi công cụ (`Action`) và nhận kết quả từ môi trường (`Observation`), giúp Agent lập kế hoạch động và tự sửa sai dựa trên dữ liệu thực.",
            "options": [
                ("Train -> Evaluate -> Deploy", False),
                ("Thought -> Action -> Observation (lặp lại tới khi có câu trả lời cuối)", True),
                ("Tokenize -> Embed -> Softmax", False),
                ("Encode -> Decode -> Quantize", False),
            ],
        },
        {
            "content": "Rủi ro bảo mật nghiêm trọng nhất khi AI Agent đọc nội dung tài liệu/web bên ngoài rồi thực thi Tool Calling có quyền ghi/xóa là gì?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "**Indirect Prompt Injection**: kẻ tấn công giấu chỉ thị độc hại trong tài liệu/trang web mà Agent đọc; nếu không sanitize, phân tách dữ liệu khỏi chỉ thị hệ thống và kiểm tra quyền/human-in-the-loop, Agent có thể bị lừa gọi tool nguy hiểm.",
            "options": [
                ("Cross-Site Scripting (CSS injection) trên font chữ", False),
                ("Indirect Prompt Injection khiến Agent thực thi tool ngoài ý muốn", True),
                ("Model bị tăng nhiệt độ GPU quá 100°C", False),
                ("JSON schema tự động chuyển sang XML", False),
            ],
        },
        {
            "content": "Giao thức **MCP (Model Context Protocol)** giải quyết bài toán gì khi xây dựng ứng dụng AI Agent?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "easy",
            "explanation": "MCP cung cấp chuẩn giao tiếp thống nhất (tương tự cổng USB-C cho AI) giữa ứng dụng LLM/Agent (client) và các nguồn dữ liệu/công cụ bên ngoài (MCP servers: resources, tools, prompts).",
            "options": [
                ("Thay thế TCP/IP ở tầng mạng vật lý", False),
                ("Chuẩn hóa cách LLM/Agent kết nối với tools, resources và context bên ngoài", True),
                ("Nén trọng số mô hình Transformer khi huấn luyện", False),
                ("Tạo giao diện CSS cho ứng dụng React", False),
            ],
        },
        {
            "content": "Vì sao khi định nghĩa Tool/Function Calling cho LLM cần dùng **Structured Outputs / Strict JSON Schema** và validate lại phía backend bằng Pydantic?",
            "code_snippet": "class SearchArgs(BaseModel):\n    query: str = Field(min_length=1, max_length=200)\n    top_k: int = Field(default=5, ge=1, le=20)",
            "code_language": "python",
            "difficulty": "medium",
            "explanation": "LLM output vẫn là đầu vào không đáng tin cậy (untrusted input). Strict JSON Schema giúp model sinh đúng định dạng, còn validate lại bằng Pydantic ở backend chặn giá trị vượt biên, thiếu trường hoặc injection trước khi gọi API/DB.",
            "options": [
                ("Vì LLM không bao giờ mắc lỗi cú pháp JSON", False),
                ("Vì đầu ra của LLM là untrusted input, cần schema strict + validate phía server trước khi thực thi tool", True),
                ("Để giảm số lượng CPU core cần thiết", False),
                ("Vì Pydantic tự động thực thi câu lệnh SQL hộ backend", False),
            ],
        },
        {
            "content": "Trong thiết kế Multi-Agent System, mô hình **Supervisor / Orchestrator** khác với mô hình **Swarm / Peer-to-Peer Handoff** ở điểm nào?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "medium",
            "explanation": "Mô hình Supervisor có một agent điều phối trung tâm phân việc và tổng hợp kết quả từ workers; mô hình Swarm/Handoff cho phép các agent chuyên biệt bàn giao (handoff) quyền xử lý trực tiếp cho nhau theo ngữ cảnh.",
            "options": [
                ("Supervisor không sử dụng LLM; Swarm không có system prompt", False),
                ("Supervisor điều phối tập trung qua 1 node trung tâm; Swarm/Handoff chuyển giao trực tiếp giữa các agent", True),
                ("Swarm chỉ chạy được tối đa 2 agent", False),
                ("Supervisor bắt buộc mọi agent dùng chung một tool duy nhất", False),
            ],
        },
        {
            "content": "Khi đánh giá (evaluate) một AI Agent thực hiện tác vụ nhiều bước, những nhóm chỉ số nào quan trọng nhất ngoài câu trả lời cuối?",
            "code_snippet": None,
            "code_language": None,
            "difficulty": "hard",
            "explanation": "Cần đánh giá cả **Task Completion / Outcome**, **Trajectory / Tool-Call Accuracy** (gọi đúng tool, đúng tham số, không lặp vô hạn), **Latency & Token Cost**, và **Safety / Policy Compliance**.",
            "options": [
                ("Chỉ cần đo BLEU score của câu trả lời cuối", False),
                ("Task success rate, trajectory/tool-call accuracy, step/token efficiency và safety guardrails", True),
                ("Chỉ cần đếm số dòng code của Agent", False),
                ("Chỉ cần kiểm tra kích thước ảnh avatar của chatbot", False),
            ],
        },
    ],
}


async def seed_database(db: AsyncSession) -> None:
    existing_cat = await db.scalar(select(Category.id).limit(1))
    if existing_cat is not None:
        return

    slug_to_category: dict[str, Category] = {}
    for cat_data in SEED_CATEGORIES:
        cat = Category(**cat_data)
        db.add(cat)
        slug_to_category[cat.slug] = cat
    await db.flush()

    for slug, questions_data in SEED_QUESTIONS.items():
        cat = slug_to_category[slug]
        quiz = Quiz(
            user_id=None,
            category_id=cat.id,
            title=f"{cat.name} Mastery Quiz",
            description=f"Bài kiểm tra chuẩn AI Engineer cho chủ đề {cat.name}.",
            difficulty="mixed",
            time_limit_seconds=600,
            is_public=True,
        )
        db.add(quiz)
        await db.flush()

        for q_idx, q_item in enumerate(questions_data):
            q = Question(
                category_id=cat.id,
                user_id=None,
                content=q_item["content"],
                code_snippet=q_item["code_snippet"],
                code_language=q_item["code_language"],
                explanation=q_item["explanation"],
                difficulty=q_item["difficulty"],
                source="seed",
                is_public=True,
            )
            for opt_idx, (opt_text, is_corr) in enumerate(q_item["options"]):
                q.options.append(
                    QuestionOption(
                        content=opt_text,
                        is_correct=is_corr,
                        order_index=opt_idx,
                    )
                )
            db.add(q)
            await db.flush()
            db.add(QuizQuestion(quiz_id=quiz.id, question_id=q.id, order_index=q_idx))

    await db.commit()

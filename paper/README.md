# Abstract

Mô hình Diffusion là rất tốt với bài toán text-to-image. Tuy nhiên với image-to-text vẫn chưa có nhiều nghiên cứu sử dụng diffusion so với các mô hình Auto-Regressive (AR)

Nghiên cứu này tập trung chỉ ra những điểm mạnh của các mô hình diffusion trong việc lập bối cảnh tổng thể và giải mã song song. Khắc phục những hạn chế của AR bao gồm: tốc độ suy luận chậm, lan truyền lỗi (error propagation), hạn chế đơn hướng (unidirectional constraints)

Nghiên cứu này còn xác định sự kém hiệu quả của các nghiên cứu trước của các mô hình diffusion xuất phát từ việc thiếu một latent space hiệu quả để căn chỉnh image-text. Và sự khác biệt giữa quá trình difusion liên tục và dữ liệu văn bản rời rạc

Nghiên cứu này đề xuất một kiến trúc mới LaDiC sử dụng một split BERT để tạo một latent space phù hợp cho việc captions và tích hợp một mô-đun regularization để  quản lý độ dài văn bản khác nhau

PHương pháp cũng bao gồm một diffuser để chuyển đổi ngữ nghĩa từ hình ảnh thành văn bản và sử dụng kỹ thuật Back&Refine để nâng cao tính tương tác của token trong quá trình suy luận.

Thực nghiệm, LaDiC cho SOTA so với các phương pháp dựa trên diffusion trên bộ dữ liệu MS COCO với 38.2 BLUE@4 và 126.2 CIDEr, mà không cần pre-training hoặc các mô-đun phụ trợ

Cho thấy khả năng cạnh tranh của các phương pháp dựa trên diffusion với các mô hình AR cho việc tạo văn bản từ hình ảnh

# Introduction

Đã có những nghiên cứu tiên phong việc sử dụng difusion vào việc sinh văn bản từ hình ảnh. Họ chủ yếu đi theo phương pháp truyền thống Encoder-Decoder, sử dụng mô hình difusion như một bộ text decoder.

Những nghiên cứu đã đưa khả năng trực quan vào mô hình bằng cách coi đâò vào hình ảnh như một loại token đặc biệt hoặc encoded hidden states, từ đó mởi rộng phạm vu nghiên cứu sang lĩnh vực multi-modal như image-to-text generation. Tuy nhiên, hiệu suất của chúng vẫn kém hơn các mô hình Auto-Regressive

Phải sử dụng kiến trúc phức tập hoặc dữ liệu thêm, các nghiên cứu dựa trên diffusion mới có thể đạt kết quả tương đương so với AR. Do đó dấy lên nghi ngờ về việc liệu các mô hình khuếch tán có những hạn chế, ít phù hợp hơn với tác vụ chuyển hình ảnh thành văn bản

Phương pháp AR tạo tuần tự các token

Các mô hình dựa trên diffusion lấy nhiễu Gausian làm đầu vào và khủ nhiễu lặp đi lặp lại theo hướng dẫn hình ảnh để tạo đồng thời toàn bộ caption

Nghiên cứu này, tác giả muốn xóa tan nhận định trên và chỉ ra những lợi ích của mô hình dựa trên diffusion so với AR:

- Parallel Decoding

![alt text](image.png)

- Holistic Context Consideration: Xem xét bối cảnh toàn diện, đa hướng, giảm tích lũy lỗi. So với BLIP, mô hình của tác giả vẫn hoạt động tốt (BLUE metric) khi chiều dài văn bản được sinh ra tăng lên

![alt text](image-1.png)

- Flexible Generation: Sinh linh hoạt, trong khi AR phải tuân thủ cách sinh một chiều cố định, có thể tùy chỉnh tạo captions dựa trên token ở mọi ví trí, trong khi đây là một thách thức đối với các mô hình AR để image captioning

![alt text](image-2.png)

Khi xem xét, tác giả đã suy ra rằng hiệu suất kém của các mô hình diffusion xuất phát từ 2 yếu tố:

- khoảng trống ngữ nghĩa trong việc dịch từ hình ảnh sang văn bản
    - khỏang cách giữa thông tin hình ảnh và biểu diễn văn bản
    - khoảng cách giữa ngữ nghĩa văn bản cấp cao và các từ cụ thể
![alt text](image-3.png)

- Sự không tương tích giữa công nghệ continuous diffusion (image generation) and đầu vào rời rạc (tạo văn bản): các mô hình continuous diffusion cổ điển thì phù hợp với không gian pixel nhưng gặp khó khăn khi chuyển trực tiếp sang không gian văn bản rời rạc. Ngoài ra, hình ảnh được tạo có kích thước cố định, trong khi độ dài caption lại khác nhau, đặt ra một thách thức cho các mô hình phổ biến trong việc xác định ranh giới của caption được tạo 

LaDiC (Latent Diffusion-based Captioner), thay vì trực tiếp tạo văn bản rời rạc từ biểu diễn hình ảnh, tác giả coi bộ diffusion như một giao diện dịch thông tin hình ảnh sang biểu diễn văn bản cấp cao (sentence latent). Cách tiếp cận này giảm bớt gánh nặng của mô hình diffusion, cho phép nó tận dụng khả năng sinh mạnh mễ của mình trong không gian ngữ nghĩa cấp cao

sau đó, sử dụng một Non-Auto-Regressive (NAR) text decoder để tạo các token rời rạc từ không gian tiềm ẩn (latent space)

Để giải quyết các vấn đề như độ dài văn bản thay đổi, tác giả đề xuất một mô-đun post-processing bao gồm normalization và reassignment

Trong quá trình suy luận, tác giả giới thiệu kỹ thuật Back&Refine để cung cấp nhiều tương tác hơn giữa các tokens để có hiệu suất tốt hơn.

# Related Works

## Diffusion Models and their Applications

foundational architectures DDPM (Ho et al 2020b) and DDIM (Song et al. 2020)

Các phương pháp này dần dần chuyển đổi các mẫu thành nhiễu Gausian và huấn luyện mô hình để phục hồi chúng, đưa ra mục tiêu học tập đơn giản và ổn định để giải quyết các vấn đề như posterior và mode collapse, vốn thách thức các mô hình trước đó như VAE và GAN

Có nhiều ứng dụng: hình ảnh, âm thanh, video, 3D, hình đại diện của con người (human avatar). Tuy nhiên ứng dụng với văn bản vẫn ở trạng thái ban đầu. Làm thế nào để điều chỉnh các token rời rạc vào một mô hình diffusion vẫn là một thách thức. Các phương pháp tiếp cận hiện tại để giải quyết vấn đề này thường thuộc 2 loại:

- Discrete Text Diffusion Models: bắt chước quá trình khuếch tán trên không gian rời rạc bằng cách làm hỏng trực tiếp văn bẳn bằng [MASK] token

- Continuous Text Diffusion Models: sử dụng các continuous embeddings để thể hiện từng token và sau đó thực hiện quy trình diffusion cổ điển. Cách tiếp cận này chứng minh tính khả thi của việc áp dụng mô hình diffusion để tạo văn bản và thể hiện khả năng so sách với các phương pháp AR, chúng bị giới hạn ở các biểu diễn đơn phương thức và có thể bỏ qua ngữ nghĩa tổng thể cấp cao. Tuy nhiên trong nghiên cứu của Lewis et al 2019, mô hình diffusion được thiết kế để dự đoán các trạng thái ẩn của BART, vẫn dựa vào cơ chế AR, gặp phải vấn đề như low inference eficiency

## Image-to-text Generation

Mục đích của Image captioning là mô tả nội dung của hình ảnh bằng ngôn ngữ tự nhiên

Có những biến thể khác như:

- densee captioning: minh họa từng đối tượng trong hình ảnh

- paragraph captioning: tạo ra một đoạn văn dài và chi tiết

Các phương pháp AR ban đầu cho captioning sử dụng kiến trúc encoder-decoder với một CNN để mã hõa ảnh và RNN để tạo ra captions. Với sự ra đời của Transformer và phương pháp large-scale pretraining, mô hình pretrained vision-language đã nổi lên và đạt hiệu suất cao

Ngược lại với việc tạo ra các mô hình AR một chiều, các mô hình NAR tạo ra toàn bộ captions song song: 

- MNIC (Gao et al., 2019) đã giới thiệu chiến lược mask token, và NAIC (Guo et al., 2020) đã sử dụng phương pháp học tăng cường trong việc tạo chú thích NAR. 

- Hầu hết các mô hình diffusion đều đi theo phương pháp mô hình continuous diffusion. 

- Ngoài ra, Bit Diffusion (Chen et al. 2022a) mã hóa captions thành bit nhị  phân và - 

- DD-Cap (Zhu et al., 2022) áp dụng mô hình discrete diffusion cho captioning.

- SCD-Net (Huo et al. 2022) là mô hình dựa trên diffusion tiên tiến nhất với một quy trình semantic-conditional diffusion (khuếch tán có điều kiện ngữ nghĩa), tuy nhiên kiến trúc xếp tầng của nó tương đối phức tạp và yêu cầu một mô-đun truy xuất bên ngoài, hạn chế khả năng mở rộng thêm của nó.

- Tác giả đề xuất một kiến trúc mới, nhỏ gọn, hiệu suất được cải thiện

# Methodology

## Overview

![alt text](image-4.png)

sử dụng một bộ text encoder để chuyển đổi không gian văn bản rời rạc C thành không gian tiềm văn bản liên tục X

Diffuser được huấn luyện để đóng vai trò là cầu nối giữa không gian biểu diễn hình ảnh V và không gian văn bản X, và cuối cùng, một bộ text decoder ánh xạ các text latent codes trở lại thành văn bản rời rạc

### Training Procedure

image v là đầu vào, sử dụng một mạng encoder để encode v thành key và Value làm đầu vào cho diffuser (thông tin có điều kiện của v)

caption c là nhãn, sử dụng một encoder đẫ được huấn luyện để encode c thành x0, thêm t lần nhiễu vào x0 rồi sử dụng xt làm đầu vào cho diffuser

diffuser tương đương với ánh xạ này: ![alt text](image-5.png)

### Inference Procedure

xt được thay thế với x∞ nhiễu Gausian thuần túy, ![alt text](image-6.png)

diffuser: ![alt text](image-7.png), x mũ 0 biểu diễn cho text latent code dự đoán

cuối cùng decoder chuyển đổi mã tiềm ẩn được dự đoán thành văn bản rời rạc c mũ

## Latent Space Tailored for Text

không gian tiềm ẩn văn bản X đóng vai trò là cầu nối giữa không gian hình ảnh V và không gian văn bản rời rạc C, giảm bớt đáng kể gánh nặng cho mô hình diffusion. Vì vậy cần thiết kế X một cách phù hợp, không gian này cần có mật độ ngữ nghĩa thích hợp để tạo thuận lợi cho việc chuyển đổi ngữ nghĩa từ hình ảnh sang văn bản

Nói chung, không gian tiềm ấn văn bản X được xây dựng bởi bộ text encoder. Thường có hai cách:

- Very shallow encoder, ví dụ một single embedding layer of BERT để chuyển đổi văn bản rời rạc thành dạng liên tục. Tuy nhiên phương pháp này thiếu sự tương tác giữa các tokens và mô hình ngữ nghĩa tổng thể, đặt ra thách thức khi khớp hình ảnh với các phần token embedding độc lập.

- sử dụng toàn bộ mô hình BERT làm bộ mã hóa văn bản, mang lại các biểu diễn ngữ nghĩa dày đặc cho các câu. Tuy nghiên, so với câu tiềm ẩn có mật độ thông tin cao, hình ảnh thường có mật độ thông tin thấp hơn nhiều, được đặc trưng bởi sự dư thừa đáng kể trong dữ liệu pixel. Sự khác biệt giữa mật độ thông tin của hình ảnh và văn bản cản trở khả năng của bộ khuếch tán trong việc học cách dịch từ hình ảnh sang văn bản một cách hiệu quả. (có chứng minh không?)

Ngược lại, với hai phương pháp trên, tác giả chia đôi mô hình BERT ở giữa, sử dụng phần dưới làm text encoder, phần trên làm NAR text decoder. Việc đặt không gian tiềm ẩn của văn bản dựa trên lớp giữa của BERT mang lại sự liên kết được cải thiện giữa hình ảnh và ngôn ngữ, từ đó nâng cao hiệu suất

Ngoài ra, để cải thiện khả năng tái tạo không gian văn bản của bộ decoder, tác giả làm cho các tham số trong language model head có thể huấn luyện được (sẽ ra sau nếu huấn luyện tất cả tham số của decoder ?)

Để đạt được một không gian đặc trưng câu được tiêu chuẩn hóa hơn có lợi cho việc thêm nhiễu, tác giả sử dụng phép chuẩn hóa tác động lên không gian này. Tác giả thu thập một tập hợp con của tất cả các captions từ tập dữ liệu và tính toán giá trị trung bình và độ lệch chuẩn của các latent codes tương ứng ![alt text](image-8.png). Trong quá trình huấn luyện, các số liệu thống kê này được sử dụng để chuẩn hóa không gian đặc trưng của BERT bằng cách thao tác trên từng mẫu như sau: ![alt text](image-9.png) (có chứng minh hiệu quả của việc chuẩn hóa không, có các làm khác không?)

thay đổi x để thông tin của các token đặc biệt không bị lẫn vào x và để diffuser dễ học ra kết thúc của caption: ![alt text](image-10.png). Trong lúc suy luận, [PAD] có thể dễ dang bị xóa bằng cách post-processing để tạo ra các chú thích có độ dài khác nhau (có chứng minh tính hiệu quả không?)

## Diffuser Mapping Image to Text

![alt text](image-11.png)

## Back&Refine Technique during Inference

Khác với các mô hình AR vốn có sự phụ thuộc tuần tự rõ ràng giữa các tokens, mô hình diffusion cho ra tất cả các tokens một cách song song. Một số token chẳng hạn như các đối tượng chính trong ảnh, có thể dễ dàng khôi phục. Ngược lại, mốt số token thể hiện chi tiết hình ảnh sẽ gặp khó khăn trong việc khôi phục. Do đó việc thêm nhiễu cùng scale vào token dễ  khôi phục là không hợp lý và không hiệu quả. Do đó, về mặt trực quan, chúng ta có thể tận dụng các token dễ dàng được khôi phục này làm điều kiện để hỗ trợ quá trình sàng lọc những token khó khôi phục hơn.

Theo đó, tác giả đề xuất phương pháp Back&Refine để khôi phục và tinh chỉnh các challenging token (độ tự tin dự đoán thấp).

![alt text](image.png)

coi L/2 tokens có độ tự tin cao nhất là dễ khôi phục

tái tạo L/2 khó khôi phục còn lại bằng nhiễu (complete Gaussian noise)

đặt lại t = T và bắt đầu một quá trình khử nhiễu mới để khôi phục challenging tokens based on the easier ones

Khi sử dụng phương pháp này, chúng ta quan sát được hiệu năng tăng

Nếu phương pháp này có hiệu quả, có ý tưởng nào tương tự cho kết quả tốt hơn không??

# Experiments

## Experimental Settings

### Dataset and Metrics

MS COCO Karpathy split: 

- 113287 training images

- 5000 validation images

- 5000 test images

- mỗi ảnh có 5 chú thích tham khảo

Metrics:

- BLUE@4

- CIDEr-D

- METEOR

- ROUGE-L

- SPICE

- CLIP-Score để đánh giá sự liên kết ngữ nghĩa giữa chú thích và hình ảnh được tạo

- BERT-Score để đánh giá chất lượng văn bản

### Implementation Details

encoder và decoder được đóng bằng, trừ LM-head. Trọng số của encoder và decoder được lấy từ 6 layer đầu tiên và 6 layers cuối cùng của BERT-base.

cơ sở lý luận để chọn không gian tiềm ẩn:


## Quantitative Analysis

## Case Study and Human Evaluation

## Unleashing the Speed of Diffusion Model

## Customizing the Generation Process

## Analysis for the Back&Refine Technique

## Ablation study


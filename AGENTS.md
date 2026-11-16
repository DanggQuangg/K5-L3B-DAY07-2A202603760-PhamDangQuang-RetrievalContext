# Herdr Multi-Agent & Skill Routing Protocol

Bạn (Antigravity - Gemini 3.8 Flash High) là **Lead Orchestrator** chạy trong Herdr. Quota của bạn rất dồi dào, trong khi các agent bên Codex (`coder-sol`, `arch-astra`, `worker-terra`) có giới hạn quota nghiêm ngặt.

## Quy tắc bắt buộc khi phối hợp qua `herdr`:
1. **Kiểm tra danh sách Agent:** Chạy `hc ls` (hoặc `herdr agent list`) để nắm rõ tên, address (`t1`, `t5`, ...) hoặc trạng thái của các agent đang mở. Không bao giờ tự gọi vào terminal của chính mình (`$HC_TERMINAL_ID` / `$HERDR_PANE_ID`).
2. **Trinh sát trước khi gọi (Scout First):** Trước khi gửi bất kỳ prompt nào cho Codex, bạn PHẢI tự tìm kiếm trong codebase, đọc các file liên quan và xác định chính xác đường dẫn file + tên hàm/số dòng.
3. **Định tuyến & Chưng cất Skill (Skill Pre-selection):**
   - Luôn kiểm tra thư mục `.agents/skills/` để tìm skill phù hợp với yêu cầu.
   - Tự đọc file `SKILL.md` của skill đó, trích xuất các quy tắc/ràng buộc quan trọng nhất liên quan đến task hiện tại và đính kèm trực tiếp vào prompt gửi cho Codex (hoặc chỉ định rõ tên skill duy nhất mà Codex cần dùng).
4. **Cấu trúc Prompt gửi cho Codex (`hc send <target> "<prompt>"` hoặc `herdr agent prompt <target> "<prompt>"`):**
   Luôn đóng gói theo cấu trúc 4 phần ngắn gọn:
   - `[TARGET FILES]`: Danh sách đường dẫn file cụ thể (yêu cầu Codex KHÔNG tự quét toàn bộ repo).
   - `[CONTEXT SUMMARY]`: Tóm tắt cấu trúc liên quan và chữ ký hàm (function signatures).
   - `[SKILL RULES]`: Các quy tắc cốt lõi đã trích xuất từ Skill tương ứng.
   - `[TASK]`: Yêu cầu thực thi rõ ràng, dứt khoát.
5. **Tự xử lý lỗi nhỏ (Self-Healing QA):** Sau khi Codex sửa file xong, bạn tự chạy build/test/linter (hoặc đọc lại qua `hc read <target>`). Nếu có lỗi syntax, import, hoặc type error đơn giản, hãy TỰ SỬA. Chỉ gọi lại `coder-sol` hoặc `arch-astra` nếu gặp lỗi logic phức tạp sau 2 lần thử.

"""Korean and English interface text (module C)."""

STRINGS = {
    "title": {"en": "Creating an ID Photo", "ko": "증명사진 만들기"},
    "live": {"en": "webcam · live", "ko": "웹캠 · 실시간"},
    "pill_relit": {"en": "Relit · live", "ko": "보정 · 실시간"},
    "pill_clothing": {"en": "Clothing · waiting for FBA", "ko": "Clothing · FBA 대기 중"},
    "pill_fba": {"en": "FBA snapshot · {age:.1f}s old", "ko": "FBA 스냅샷 · {age:.1f}초 전"},
    "pill_modnet": {"en": "MODNet snapshot · {age:.1f}s old", "ko": "MODNet 스냅샷 · {age:.1f}초 전"},
    "pill_modnet_waiting": {"en": "MODNet waiting", "ko": "MODNet 대기 중"},
    "pill_before": {"en": "Before · raw", "ko": "보정 전 · 원본"},
    "hold_b": {"en": "Hold B: before", "ko": "B 누르기: 보정 전"},
    "language": {"en": "한국어", "ko": "English"},
    "format": {"en": "FORMAT", "ko": "형식"},
    "fmt_resume": {"en": "Résumé 3×4", "ko": "이력서 3×4"},
    "fmt_id": {"en": "ID 3.5×4.5", "ko": "신분증 3.5×4.5"},
    "fmt_profile": {"en": "Profile 1:1", "ko": "프로필 1:1"},
    "fmt_resume_long": {"en": "3 × 4 cm", "ko": "3 × 4 cm"},
    "fmt_id_long": {"en": "3.5 × 4.5 cm", "ko": "3.5 × 4.5 cm"},
    "fmt_profile_long": {"en": "square", "ko": "정사각형"},
    "head_size": {"en": "HEAD SIZE", "ko": "머리 크기"},
    "head_pct": {"en": "head {pct}% of height", "ko": "머리가 높이의 {pct}%"},
    "background": {"en": "BACKGROUND", "ko": "배경"},
    "bg_white": {"en": "white", "ko": "흰색"},
    "bg_light_gray": {"en": "light gray", "ko": "연회색"},
    "bg_light_blue": {"en": "light blue", "ko": "연하늘색"},
    "bg_mint": {"en": "mint", "ko": "민트색"},
    "bg_wall": {"en": "original wall", "ko": "원래 배경"},
    "matting_model": {"en": "MATTING COMPARISON", "ko": "배경 분리 비교"},
    "clothing_toggle": {"en": "Clothing improvement (off: Main)", "ko": "옷 영역 개선 (끄면 Main)"},
    "clothing_toggle_tip": {"en": "On: Clothing. Off: Main baseline.",
                            "ko": "켬: Clothing, 끔: Main 기준선."},
    "fba_demo": {"en": "FBA + Clothing (noncommercial demo only)",
                 "ko": "FBA + Clothing (비상업 발표 전용)"},
    "fba_demo_tip": {"en": "Research weights; local presentation only, not for commercial use. Slow.",
                     "ko": "연구용 가중치: 로컬 발표 전용, 상업적 사용 불가. 느릴 수 있습니다."},
    "fba_waiting": {"en": "Camera: Clothing until a face and FBA snapshot are ready.",
                    "ko": "카메라: 얼굴과 FBA 스냅샷이 준비될 때까지 Clothing."},
    "fba_snapshot_age": {"en": "Camera: FBA snapshot from {age:.1f}s ago; not live video.",
                         "ko": "카메라: {age:.1f}초 전 FBA 스냅샷. 실시간 영상 아님."},
    "fba_capturing": {"en": "Rendering FBA capture…", "ko": "FBA 촬영 결과 처리 중…"},
    "fba_clothing_preview": {"en": "Clothing · live preview", "ko": "Clothing · 실시간 미리보기"},
    "fba_error": {"en": "FBA error: {err}", "ko": "FBA 오류: {err}"},
    "fba_required": {
        "en": "FBA is required but not ready: {err}\n"
              "Review the FBA authors' noncommercial terms, then run:\n"
              "  uv sync --extra fba\n"
              "  uv run --extra fba python tools/fba_reference.py --accept-noncommercial",
        "ko": "FBA가 필요하지만 준비되지 않았습니다: {err}\n"
              "FBA 저자의 비상업 이용 조건을 확인한 뒤 실행하세요:\n"
              "  uv sync --extra fba\n"
              "  uv run --extra fba python tools/fba_reference.py --accept-noncommercial"},
    "fba_unavailable": {
        "en": "FBA unavailable: {err}\nStarted in traditional Clothing mode. From a source checkout, "
              "for local noncommercial use, review the authors' terms, then run: uv sync --extra fba --group dev; "
              "uv run --extra fba python tools/fba_reference.py --accept-noncommercial",
        "ko": "FBA를 사용할 수 없습니다: {err}\n전통 Clothing으로 시작했습니다. 소스 저장소에서 로컬 비상업 사용 조건을 확인한 후 "
              "실행: uv sync --extra fba --group dev; "
              "uv run --extra fba python tools/fba_reference.py --accept-noncommercial"},
    "modnet_demo": {"en": "MODNet alpha (pretrained comparison)",
                    "ko": "MODNet 알파 (사전학습 비교)"},
    "modnet_enable_hint": {
        "en": "Enable with --presentation-modnet and a local model and dependencies.",
        "ko": "--presentation-modnet 옵션과 로컬 모델·선택 의존성이 필요합니다."},
    "modnet_demo_tip": {
        "en": "Apache-2.0 model; experimental asynchronous matting. The first frame prepares the model.",
        "ko": "Apache-2.0 모델, 실험용 비동기 매팅입니다. 첫 프레임은 모델을 준비합니다."},
    "modnet_waiting": {"en": "MODNet is preparing a fresh matte.",
                       "ko": "MODNet 새 마스크를 준비하고 있습니다."},
    "subject_no_face": {"en": "Stand in the centre and face the camera.",
                        "ko": "화면 중앙에서 카메라를 바라봐 주세요."},
    "subject_ambiguous": {"en": "Unable to separate people. Stand alone in the centre.",
                          "ko": "사람을 구분할 수 없습니다. 혼자 중앙에 서 주세요."},
    "subject_changed": {"en": "The selected person changed. Please try again.",
                        "ko": "선택된 사람이 바뀌었습니다. 다시 촬영해 주세요."},
    "modnet_snapshot_age": {
        "en": "MODNet {provider} snapshot {age:.1f}s old · frame {frame_id} · model {inference_ms:.0f} ms",
        "ko": "MODNet {provider} 스냅샷 {age:.1f}초 전 · 프레임 {frame_id} · 모델 {inference_ms:.0f} ms"},
    "modnet_capturing": {"en": "Rendering full-resolution MODNet capture…",
                         "ko": "MODNet 전체 해상도 촬영 결과 처리 중…"},
    "modnet_clothing_preview": {"en": "Clothing · live preview", "ko": "Clothing · 실시간 미리보기"},
    "modnet_main_preview": {"en": "Main · live preview", "ko": "Main · 실시간 미리보기"},
    "modnet_output_preview": {
        "en": "MODNet · live output (Clothing bypassed)",
        "ko": "MODNet · 실시간 결과 (Clothing 미사용)"},
    "modnet_error": {"en": "MODNet error: {err}", "ko": "MODNet 오류: {err}"},
    "birefnet_capture": {
        "en": "BiRefNet portrait for captures only (local, opt-in)",
        "ko": "촬영에만 BiRefNet portrait 사용 (로컬, 선택 기능)"},
    "birefnet_capture_tip": {
        "en": "Uses the verified local BiRefNet portrait model only when taking a photo. Live preview and checks stay on MODNet; failures do not fall back.",
        "ko": "사진을 찍을 때만 검증된 로컬 BiRefNet portrait 모델을 사용합니다. 실시간 미리보기와 검사는 MODNet이며 실패 시 대체하지 않습니다."},
    "birefnet_capture_unavailable": {
        "en": "BiRefNet capture unavailable: {err}",
        "ko": "BiRefNet 촬영을 사용할 수 없음: {err}"},
    "birefnet_capture_selected": {
        "en": "BiRefNet is selected for shutter captures; preview and checks remain MODNet.",
        "ko": "촬영에 BiRefNet을 선택했습니다. 미리보기와 검사는 계속 MODNet입니다."},
    "birefnet_capture_busy": {
        "en": "Rendering the BiRefNet portrait capture; MODNet preview continues independently.",
        "ko": "BiRefNet 촬영 결과 처리 중입니다. MODNet 미리보기는 별도로 계속됩니다."},
    "birefnet_capture_off": {
        "en": "Shutter captures use full-resolution MODNet.",
        "ko": "촬영 시 전체 해상도 MODNet을 사용합니다."},
    "birefnet_capture_error": {
        "en": "BiRefNet capture failed; no alternate matte was used: {err}",
        "ko": "BiRefNet 촬영 실패: 다른 마스크로 대체하지 않았습니다: {err}"},
    "finishing_processing": {"en": "Finishing the current frame before closing…",
                             "ko": "현재 프레임 처리를 마친 뒤 닫습니다…"},
    "strength": {"en": "RELIGHT STRENGTH", "ko": "조명 보정 강도"},
    "checks": {"en": "CHECKS", "ko": "점검"},
    "rule_even_light": {"en": "Even light", "ko": "고른 조명"},
    "rule_glare": {"en": "No glare", "ko": "반사 없음"},
    "rule_sharpness": {"en": "Sharp", "ko": "선명함"},
    "rule_head_pose": {"en": "Head straight", "ko": "바른 자세"},
    "rule_eyes_open": {"en": "Eyes open", "ko": "눈 뜸"},
    "rule_expression": {"en": "Mouth closed", "ko": "입 다묾"},
    "rule_clean_cutout": {"en": "Clean cutout", "ko": "깔끔한 배경 분리"},
    "msg_turn_to_light": {"en": "one side of your face is darker. Turn a little toward the light",
                          "ko": "얼굴 한쪽이 어둡거나 그림자가 졌습니다. 빛이 오는 쪽으로 조금 돌아 앉으세요"},
    "msg_glare_glasses": {"en": "your glasses reflect the light. Tilt them slightly or lower your chin a little",
                          "ko": "안경에 빛이 비칩니다. 안경을 살짝 기울이거나 턱을 조금 내리세요"},
    "msg_glare_skin": {"en": "your skin looks shiny. Blot it with a tissue or avoid strong direct light",
                       "ko": "피부가 번들거립니다. 휴지로 유분을 닦거나 강한 조명을 피해 주세요"},
    "msg_hold_still": {"en": "the photo is blurry. Hold still, and if it stays blurry, wipe the camera lens",
                       "ko": "사진이 흐립니다. 움직이지 말고, 계속 흐리면 카메라 렌즈를 닦아 주세요"},
    "msg_straighten_head": {"en": "your head is tilted to one side. Straighten it",
                            "ko": "고개가 한쪽으로 기울었습니다. 똑바로 세워 주세요"},
    "msg_face_camera": {"en": "your face is turned sideways or up or down. Face the camera straight on",
                        "ko": "얼굴이 옆이나 위아래로 돌아갔습니다. 카메라를 정면으로 바라보세요"},
    "msg_open_eyes": {"en": "your eyes look closed. Open them and look at the lens",
                      "ko": "눈이 감긴 것 같습니다. 눈을 뜨고 렌즈를 바라보세요"},
    "msg_close_mouth": {"en": "your mouth is open. Close it gently", "ko": "입이 벌어져 있습니다. 입을 다물어 주세요"},
    "msg_move_from_wall": {"en": "the edge between you and the wall is unclear. Move about 1 m away from the wall",
                           "ko": "몸과 벽의 경계가 흐릿합니다. 벽에서 1m 정도 떨어져 앉으세요"},
    "msg_top_vs_wall": {"en": "your top is close to the wall's color. Wear a top in a different color",
                        "ko": "상의 색이 벽과 비슷합니다. 벽과 다른 색의 상의를 입어 주세요"},
    "msg_plain_wall": {"en": "the room behind you is too busy to cut out cleanly. Sit in front of a plain wall",
                       "ko": "뒤 배경이 복잡해 깔끔하게 분리되지 않습니다. 단색 벽 앞에 앉아 주세요"},
    "msg_no_face": {"en": "sit in front of the camera and look at it", "ko": "카메라 앞에 앉아 화면을 바라봐 주세요"},
    "msg_move_back": {"en": "you are too close, so the photo gets cut off. Move back a little",
                      "ko": "너무 가까워서 사진이 잘립니다. 조금 뒤로 물러나세요"},
    "no_face": {"en": "No face found", "ko": "얼굴이 보이지 않습니다"},
    "framing": {"en": "Framing", "ko": "사진 틀"},
    "all_pass": {"en": "All checks pass: hold still", "ko": "모든 점검 통과: 그대로 계세요"},
    "checks_pending": {"en": "Checks not evaluated yet", "ko": "점검 결과 미판정"},
    "rule_pending": {"en": "not evaluated", "ko": "미판정"},
    "auto_shutter": {"en": "AUTO SHUTTER", "ko": "자동 촬영"},
    "passing": {"en": "{n} of {m} passing frames", "ko": "통과 프레임 {n} / {m}"},
    "auto": {"en": "Auto", "ko": "자동"},
    "take_photo": {"en": "Take photo", "ko": "지금 촬영"},
    "mirror": {"en": "Mirror view", "ko": "거울 모드"},
    "notice": {"en": "Not for passports or government IDs: they require unedited photos.",
               "ko": "여권·신분증 등 공공기관 제출용이 아닙니다: 공공기관 제출용으로는 보정하지 않은 사진을 사용하세요."},
    "crop_label": {"en": "{fmt} crop · head {pct}%", "ko": "{fmt} 크롭 · 머리 {pct}%"},
    "crown": {"en": "crown", "ko": "정수리"},
    "chin": {"en": "chin", "ko": "턱"},
    "lock_warming": {"en": "Camera warming up…", "ko": "카메라 준비 중…"},
    "lock_locked": {"en": "Exposure and white balance locked", "ko": "노출·화이트밸런스 고정됨"},
    "lock_partial": {"en": "Exposure locked (white balance cannot be locked on this camera)",
                     "ko": "노출 고정됨 (이 카메라는 화이트밸런스 고정 불가)"},
    "lock_unsupported": {"en": "Exposure lock unavailable: light averaged over 10 frames",
                         "ko": "노출 고정 미지원: 조명을 10프레임 평균으로 보정"},
    "lock_failed": {"en": "Exposure lock failed: light averaged over 10 frames",
                    "ko": "노출 고정 실패: 조명을 10프레임 평균으로 보정"},
    "lock_drifting": {"en": "Brightness drifting: light averaged over 10 frames",
                      "ko": "밝기 변동 감지: 조명을 10프레임 평균으로 보정"},
    "result_title": {"en": "Your photos", "ko": "완성된 사진"},
    "save": {"en": "Save…", "ko": "저장…"},
    "retake": {"en": "Retake", "ko": "다시 찍기"},
    "open_folder": {"en": "Open folder", "ko": "폴더 열기"},
    "saved_to": {"en": "Saved {n} files to {path}", "ko": "{path}에 파일 {n}개를 저장했습니다"},
    "sheet_note": {"en": "Print sheets (6 × 4 in, 300 DPI) are saved for the résumé and ID formats.",
                   "ko": "이력서·신분증 형식은 인화용 시트(6 × 4인치, 300 DPI)도 함께 저장됩니다."},
    "choose_folder": {"en": "Choose a folder for the photos", "ko": "사진을 저장할 폴더를 고르세요"},
    "kept_wall": {"en": "The cutout was not clean (check 7), so the original wall was kept.",
                  "ko": "배경 분리가 깔끔하지 않아(점검 7) 원래 벽을 그대로 두었습니다."},
    "camera_error": {"en": "Could not start the camera: {err}", "ko": "카메라를 시작할 수 없습니다: {err}"},
    "camera_allow": {"en": "Allow camera access in the macOS dialog. No dialog? Run the app from Terminal, "
                           "or turn on the app you started it from in System Settings → Privacy & Security → Camera.",
                     "ko": "macOS 창에서 카메라 접근을 허용해 주세요. 창이 안 뜨면 터미널 앱에서 실행하거나, "
                           "시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 이 앱을 실행한 프로그램을 켜 주세요."},
    "camera_denied": {"en": "Camera access is off for the app you started this from. Turn it on in System Settings → "
                            "Privacy & Security → Camera, then quit that app and start again.",
                      "ko": "이 앱을 실행한 프로그램의 카메라 접근이 꺼져 있습니다. 시스템 설정 → 개인정보 보호 및 보안 → "
                            "카메라에서 켠 뒤, 그 프로그램을 종료하고 다시 실행해 주세요."},
}


def t(key: str, lang: str, **kwargs) -> str:
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry["en"]
    return text.format(**kwargs) if kwargs else text

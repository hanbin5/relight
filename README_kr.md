[English](README.md) | [한국어](README_kr.md)

# Module B 옷 영역 복원 실험

![대표 사진의 원본과 옷 영역 실험 결과](docs/evaluation/clothing/research-gamma/live/comparison.jpg)

기존 배경 분리 과정이 익숙하지 않다면 먼저 [Module B 배경 분리 설명](https://github.com/hanbin5/relight/blob/module-b/matting/README_kr.md)을 읽어 주세요. 이 문서는 기존 과정을 반복하지 않고, 빠진 옷을 복원하기 위해 시도한 방법과 그 결과를 설명합니다. 변경은 오프라인 실험에만 머물지 않고 앱의 `idphoto/matting.py`에 적용되어 있습니다.

여기서 **사람 prior**는 얼굴 랜드마크로 사람이 있을 만한 위치를 가정한 것이고, **gate**는 그 위치를 제한하는 영역입니다. **색 표본 후보**는 색 모델을 만들 때 뽑을 수 있는 픽셀이며, **alpha**는 0이면 배경, 1이면 인물, 그 사이면 두 색이 섞이는 정도입니다. 앱은 옷을 의미적으로 인식하는 분할 모델을 쓰지 않습니다. SCHP·Sapiens는 오프라인 실험이며 FBA는 아래의 비상업 발표용 옵트인 데모에서도 사용할 수 있습니다. Sapiens와 FBA에는 비상업 제한이 있습니다.

## 앱에서 Main / Clothing 비교

`uv sync --group dev`와 `uv run python tools/fetch_model.py`로 환경과 얼굴 랜드마크 모델을 준비한 뒤 `uv run idphoto --lang ko`를 실행하면 카메라 화면이 열립니다. 카메라 허용이 필요하며 로컬에서만 처리합니다. 카메라 없이 확인하려면 `uv run idphoto --source demo --lang ko`, 정지 사진은 `uv run idphoto --source 사진경로 --lang ko`를 사용합니다.

오른쪽 **배경 분리 비교 → 옷 영역 개선**을 켜면 기존 Clothing 방식(기본값), 끄면 옷 추가 판정 없는 Main 방식입니다. 전환 시 색 모델을 초기화하고 자동 촬영 통과 프레임을 다시 셉니다. 촬영 결과에도 선택한 방식이 적용되며, 배경 점검 실패 시 기존 규칙대로 원래 벽을 유지합니다.

### FBA 수업 발표용 로컬 데모

FBA 코드는 MIT이지만 공개 가중치는 [저자 안내](https://github.com/MarcoForte/FBA_Matting)에 따라 **비상업적 사용만 가능**합니다. 아래 모드는 수업 발표 등 로컬 비상업 데모만을 위한 옵트인 기능이며 제품 기본값이나 가중치 배포 기능이 아닙니다. 기본 앱에는 Torch를 추가하지 않습니다. 저자의 조건을 확인한 뒤 저장소에서:

```bash
uv sync --extra fba --group dev
uv run --extra fba python tools/fba_reference.py --accept-noncommercial
uv run --extra fba idphoto --presentation-fba --lang ko
# 카메라 대신 사진: --source docs/evaluation/test/shot5.jpg
```

카메라 권한은 사용자가 직접 허용해야 합니다. `--presentation-fba`를 넣어야 **FBA + Clothing (비상업 발표 전용)**이 보이며 기본은 꺼져 있습니다. 선택하면 같은 프레임에서 Clothing alpha의 자동 트라이맵으로 FBA를 실행하고, FBA 전경색을 선형광으로 변환해 조명 비율을 적용한 뒤 합성합니다. Main은 옷 개선을 끈 상태, Clothing은 옷 개선만 켠 상태입니다. FBA 옵션은 Clothing이 켜져 있을 때만 선택할 수 있습니다. FBA 전경과 alpha는 기존 NumPy 합성과 달라 색상이 일치하지 않을 수 있습니다.

선택 의존성이나 검증된 로컬 소스·가중치가 없으면 FBA 선택이 비활성화되고 이유가 표시됩니다. 가중치 검사에 실패하면 다른 결과로 조용히 대체하지 않습니다. 영상과 사진은 외부로 보내지 않습니다.

**FBA는 실시간 영상이 아닙니다.** 기존에는 카메라 획득과 Qt UI는 별도였지만 처리 Worker가 매 프레임 FBA를 기다려 미리보기 갱신이 느려졌습니다. 이제 FBA 선택 시 큰 화면은 계속 **Clothing 실시간 미리보기**이고, 옆의 별도 이미지는 해당 프레임으로 계산한 마지막 **FBA 스냅샷**과 그 프레임의 경과 시간입니다. FBA 작업은 동시에 하나만 실행하며 대기 미리보기는 최신 한 장으로 교체합니다. 이전 프레임의 alpha를 새로운 영상에 덧씌우지 않습니다. 촬영 요청은 대기 미리보기보다 우선하고 촬영 시점의 프레임과 설정으로 전체 해상도 FBA를 별도 실행하므로 시간이 더 걸릴 수 있습니다. 촬영 중 상태가 표시되며 중복 촬영은 허용하지 않습니다.

**검사와 자동 촬영은 Clothing 미리보기 기준**입니다. 따라서 배경 점검 실패 시 원래 벽을 유지하는 촬영 fallback도 Clothing 기준이고, 과거 FBA alpha 기준 검사와 판정이 다를 수 있습니다. 최종 사진은 저장된 스냅샷이 아니라 촬영 프레임의 FBA 결과입니다. 결과가 느리거나 실패하면 상태에 표시됩니다. Main과 Clothing 단독 모드는 기존과 같습니다.

참고 계측: Apple M4 Pro, 같은 shot3의 640×360 프레임·캐시 랜드마크, strength 0, 실제 로컬 FBA 가중치(타일 768, 겹침 96, Torch 4 threads). 카메라·랜드마크 검출·화면 렌더링은 제외했습니다. 변경 전 단일 처리 경로에서 8회 warm 평균은 Main 33.2 ms, Clothing 36.3 ms, FBA + Clothing 273.1 ms(3.66 fps), FBA 첫 프레임 684.5 ms였습니다. 변경 후 오프스크린 Worker에 같은 프레임을 최대 30 fps로 공급하고 첫 2초를 제외한 8초를 측정한 결과 Main 27.0 fps, Clothing 25.75 fps, FBA 선택 시 Clothing 화면 22.125 fps와 FBA 스냅샷 4.0 fps, 스냅샷 평균 경과 0.335초였습니다. 별도 추론도 CPU를 공유하므로 미리보기가 항상 부드럽다는 보장은 없습니다. 이는 실제 카메라/네이티브 화면 측정이 아니며 환경과 입력 해상도에 따라 달라집니다. 실제 네이티브 앱 창·카메라 동작은 이 환경에서 검증하지 못했습니다.

### FBA + Clothing 옆 배경 누출 진단

아래 네 실험의 비교 이미지는 인물이 포함되어 저장소에 올리지 않습니다. 로컬 평가 명령으로 생성할 수 있습니다.

[라이브 CSV](docs/evaluation/clothing/fba-leakage/live/measurements.csv) · [전체 격자 CSV](docs/evaluation/clothing/fba-leakage/full-grid/measurements.csv)

신고된 실제 카메라 프레임은 저장되지 않아, 같은 문제일 가능성을 기존 12장으로 확인했습니다. 640 px 입력, 같은 랜드마크, 사진별 새 `Matter`, strength 0과 앱 발표 모드의 FBA 타일 768·겹침 96을 사용했습니다. 라이브는 `s=2`, 전체 격자는 같은 640 px 입력의 `s=1`이며 연속 카메라의 warm state나 실제 앱 화면을 재현한 것은 아닙니다. FBA 합성은 발표 모드처럼 선형광 흰 배경을 사용하며, 이 전의 1024 타일·sRGB 합성 그림과 수치가 다를 수 있습니다.

Clothing 자체의 test1 결과에는 이미 옷 옆 어두운 물체가 남습니다. 자동 트라이맵은 그 alpha를 침식한 안쪽을 확정 전경으로, 넓어진 마스크 주위를 미확정으로 만듭니다. 단계별 그림의 빨강은 Main 미확정 → Clothing 확정 전경, 파랑은 Main 확정 배경 → Clothing 미확정, 보라는 Main 확정 배경 → Clothing 확정 전경입니다. 이 변화는 **옷과 배경이 섞인 전체 영역**이므로 모두 오류라는 뜻이 아닙니다. 예를 들어 라이브 test1은 각각 24,014·9,824·0픽셀이 바뀝니다. 확정 전경으로 지정된 배경은 FBA가 제거할 수 없고, 미확정 영역에서도 test4의 Main FBA처럼 옷 주변 배경이 흐리게 남을 수 있습니다. 따라서 Clothing만의 문제나 FBA만의 문제로 단정할 수 없습니다.

고정 후보 **Guarded**는 Main 트라이맵의 확정 배경을 보존하고, Main 미확정 중 Clothing 확정 전경이면서 첫 Main FBA alpha > 0.75인 곳만 확정 전경으로 승격해 FBA를 다시 실행합니다. 기준 0.75는 이전에 이 사진들을 보며 정한 값입니다. 라이브 test1에서 Clothing FBA 대비 alpha 0.5 초과가 11,701픽셀 줄어도 옷과 배경이 함께 포함되며, 옆 배경은 여전히 보입니다. shot5에서는 같은 비교의 31,706픽셀이 줄면서 실제 재킷 양쪽도 크게 사라집니다. 머리와 옷을 보존하면서 배경만 안정적으로 제거한 근거가 없어 이 후보는 발표 앱에 채택하지 않았습니다. 앞서 시도한 형태·연결·밝기 경계 후보도 아래 실패 사례를 참고하세요.

CSV의 픽셀 수와 alpha 차이는 정답 오류율이 아니며, 정답 마스크 없는 12장으로 일반 정확도 우위를 주장할 수 없습니다. 사진은 로컬에서만 처리합니다. 아래 재현에는 기존 로컬 랜드마크 캐시와 FBA 자산이 필요합니다.

```bash
uv run --extra fba --group dev python -m tools.eval_fba_leakage --out out/fba-leakage/live
uv run --extra fba --group dev python -m tools.eval_fba_leakage --full-grid --out out/fba-leakage/full-grid
```

### 색 근거로 옆 배경을 제한하는 추가 실험

[라이브 CSV](docs/evaluation/clothing/fba-color-constraints/live/measurements.csv) · [전체 격자 CSV](docs/evaluation/clothing/fba-color-constraints/full-grid/measurements.csv)

같은 12장과 발표 모드의 타일 설정으로 사진별 조정 없이 네 후보를 비교했습니다. 기존 Clothing 트라이맵의 확정 배경과 턱 아래 중앙 확정 전경에서 각각 최대 2,000픽셀을 표본으로 뽑아 기존 sqrt-linear 색 거리와 k-means를 사용합니다. 중앙 전경 표본은 턱 + 0.25 `ec` 아래, 중심선 ±0.65 `fw`입니다. 턱 + 0.2 `ec` 아래에서 배경 거리 < 1.0, 중앙 옷 거리 > 2.5인 픽셀을 배경 근거로 봅니다. 이는 확실한 배경 라벨이 아닙니다. 두 표본 중 하나가 50픽셀 미만이면 후보를 적용하지 않고 CSV에 표시합니다.

- **Color BG trimap:** 배경 근거와 겹치는 미확정 영역을 확정 배경으로 바꾸고 FBA를 새로 실행합니다.
- **BG + FG demotion:** 위 변경에 더해 배경 근거와 겹치는 확정 전경은 미확정으로 낮춘 후 FBA를 다시 실행합니다.
- **Buffered BG trimap:** 첫 Clothing FBA alpha > 0.8 또는 원래 확정 전경을 얼굴 너비의 0.25배만큼 팽창해 보호합니다. 보호 영역 밖의 미확정 배경 근거만 확정 배경으로 바꿔 FBA를 다시 실행합니다.
- **Color soft post:** 원래 FBA의 미확정 영역 중 배경 근거에 해당하는 alpha만 절반으로 낮춥니다. 전경색은 다시 추정하지 않습니다.

라이브 test3·test4의 배경 안개는 보호 없는 트라이맵 후보에서 줄지만 어두운 물체 일부는 남습니다. shot5 재킷은 대체로 남지만, **shot3은 화면 왼쪽 재킷이 보호 없는 트라이맵 후보에서 더 잘립니다.** Buffered는 shot3 재킷을 더 남기지만 test3·test4의 옆 배경도 상당히 남깁니다. 후처리 후보는 test3·test4의 배경에 수평 경계가 드러납니다. test1·test2는 색 근거가 0픽셀이라 네 후보 모두 누출을 해결하지 못합니다. 머리 위의 입력 트라이맵은 바꾸지 않아도 FBA 재실행으로 머리 결과가 바뀔 수 있습니다.

또한 정답이 있는 합성 재킷의 한 소매를 벽과 같은 색으로 만든 반례에서 4,249개 소매 픽셀 모두가 배경 근거로 판정되어 보호 없는 트라이맵 후보의 확정 배경이 됩니다. 기존 FBA 역시 이 합성 소매의 alpha를 이미 낮게 추정하므로 이는 이번 후보가 기존 결과를 4,249픽셀만큼 추가로 손상했다는 뜻이 아니라, 색만으로는 옷 보존을 보장하지 못한다는 확인입니다. 옷 손실과 잔여물 때문에 네 후보 모두 앱에 적용하지 않았습니다. 실제 카메라 프레임은 없고 사진에 정답 마스크도 없으므로 이 결과를 실제 카메라의 재현이나 일반적인 정확도 개선으로 해석할 수 없습니다.

```bash
uv run --extra fba --group dev python -m tools.eval_fba_color_constraints --out out/fba-color-constraints/live
uv run --extra fba --group dev python -m tools.eval_fba_color_constraints --full-grid --out out/fba-color-constraints/full-grid
```

## 사람 형태와 밝기 경계 prior: 오프라인 비교

[라이브 CSV](docs/evaluation/clothing/shape-priors/live/measurements.csv) · [전체 격자 CSV](docs/evaluation/clothing/shape-priors/full-grid/measurements.csv)

아래 FBA 조합과 같은 12장·640 px 입력 및 같은 캐시 랜드마크를 쓰고, 사진별 조정 없이 다음 후보를 시험했습니다. 자세나 가려짐을 추정하지 못하므로 사람 밖을 확정 배경으로 자르지 않습니다. 기존 옷 모듈의 `clothing_regions` 몸통 prior를 `(likely - 0.5) / 0.2`로 0–1에 제한한 값 `T`와 턱에서 아래로 `0.4 × 눈–턱 거리`에 걸쳐 0에서 1로 증가하는 값 `B`를 씁니다. 연결성 `C`는 **옷 트라이맵 FBA alpha > 0.5**의 8-연결 성분 중 얼굴 마스크와 겹치는 성분입니다. 연결된 배경도 포함될 수 있고, 끊어진 옷도 제외될 수 있습니다.

- **Shape + connect trimap / FBA:** 옷 트라이맵의 확정 전경 중 Main 트라이맵은 미확정이고 `B ≥ 0.5`이며 `T < 0.5` 또는 얼굴 비연결인 픽셀만 미확정으로 낮춘 뒤 FBA를 새로 실행합니다. 확정 배경을 새로 지정하지 않습니다.
- **Shape post:** 옷 트라이맵 FBA 결과에서 Main 트라이맵이 미확정인 픽셀의 alpha에 `1 - 0.2 × B × (1 - T)`를 곱합니다.
- **Shape + connect post:** 위 감소에 `0.15 × B × (1 - C)`를 더합니다. 한 픽셀의 alpha를 최대 35%만 낮추며, Main 트라이맵의 확정 픽셀은 기존 옷 기반 FBA 값을 그대로 보존합니다. Main의 확정 배경을 새 결과에 강제로 적용하는 것은 아닙니다.
- **Shape + connect + edge post:** 위 감소에 선형광 휘도 Sobel 경사 `g`의 `1 - clip(g / 0.04, 0, 1)`을 곱해 강한 밝기 경계에서 감쇠를 줄입니다. 전경색은 FBA 추정값을 재사용합니다.

새 트라이맵은 라이브 12장 중 5장, 전체 격자 6장에서 바뀌지 않아 재실행 결과도 사실상 기존 결합과 같습니다. 라이브 test2에서는 확정 전경 424픽셀을 미확정으로 바꿨지만 주변 배경이 여전히 남습니다. 후처리는 test1·test4의 배경과 옷을 함께 옅게 만들 수 있고, test9에서는 인물과 연결된 벽·그림자가 남습니다. shot5에서는 기존 결합의 재킷은 보존되지만 후처리로 일부가 옅어집니다. 강한 휘도 경계가 옷 경계라는 보장은 없어 edge 후보도 그림자나 무늬를 보호할 수 있습니다. 유의미한 안정적 개선을 확인하지 못해 어느 후보도 앱에 채택하지 않았습니다.

CSV의 추가·제거는 `FBA: clothing`과 alpha 0.5 기준을 비교한 전체 픽셀 수이며 정답 오류나 옷 복원량이 아닙니다. 예를 들어 shape + connect 후처리에서 라이브 test1은 1,216개, test4는 1,906개가 0.5 이하로 바뀌지만 옷과 배경이 섞여 있습니다. 12장에 정답 마스크가 없고 관찰이 독립 검증도 아니므로 정확도 우위를 주장할 수 없습니다. 앞선 결합과 마찬가지로 FBA는 비상업적 로컬 연구로만 실행했습니다.

```bash
uv run --extra fba --group dev python -m tools.eval_shape_priors --out out/shape-priors/live
uv run --extra fba --group dev python -m tools.eval_shape_priors --full-grid --out out/shape-priors/full-grid
```

### 조명·깊이 관련 연구와 이번 후보의 차이

- [Virtual Multi-Modality Self-Supervised Foreground Matting for Human-Object Interaction](https://arxiv.org/abs/2110.03278) (ICCV 2021)은 RGB 한 장에서 깊이·분할·상호작용 지도를 추정해 사람과 상호작용 물체를 함께 매팅합니다. 단안 깊이 추정의 실제 사례지만 사람만 남기는 증명사진 문제와 목적이 다릅니다.
- [DART: Depth-Enhanced Accurate and Real-Time Background Matting](https://arxiv.org/abs/2402.15820) (arXiv 2024)은 **RGB-D 카메라**의 깊이와 배경 깊이 prior로 트라이맵을 정제하며 정적 배경을 전제로 합니다. RGB 한 장에서 깊이를 추정하는 방법은 아닙니다.
- [Total Relighting: Learning to Relight Portraits for Background Replacement](https://augmentedperception.github.io/total_relighting/) (SIGGRAPH 2021)은 매팅 후 목표 조명에 맞춰 인물을 재조명해 합성합니다. 조명 정보를 매팅 prior로 쓰는 사례는 아닙니다. [Controllable Light Diffusion for Portraits](https://arxiv.org/abs/2305.04745) (CVPR 2023)는 인물의 강한 그림자·반사를 완화해 분할 같은 후속 작업에 활용하는 연구이며 매팅 입력 prior 자체는 아닙니다.

이번 휘도 경사는 원본 선형 밝기의 국소 변화만 이용하며 **조명 또는 깊이를 추정하지 않습니다**. 사용 가능한 기존 실험 자산에는 깊이 모델이나 깊이 지도가 없고, 별도의 깊이 모델은 가중치·라이선스와 계산량 검토가 필요하므로 다운로드하거나 적용하지 않았습니다. 위 두 조명 논문은 조명 처리가 관련될 수 있음을 보여 주지만, 조사한 출처만으로 조명 추정을 직접 인물 매팅 prior로 쓴다고 주장하지 않습니다.

## FBA와 옷 모듈 조합 비교

[라이브 CSV](docs/evaluation/clothing/fba-combinations/live/measurements.csv) · [전체 격자 CSV](docs/evaluation/clothing/fba-combinations/full-grid/measurements.csv)

두 모드 모두 640 px 너비 입력이며 각각 라이브 `s=2`, 촬영용 전체 격자 `s=1`입니다. 전체 격자는 원본 해상도가 아닙니다. 같은 사진별 랜드마크와 relighting strength 0을 쓰고 각 사진마다 새로운 `Matter`를 사용했습니다. 자동 트라이맵은 alpha > 0.5 마스크를 얼굴 너비의 0.06배만큼 침식한 전경과 얼굴 마스크를 확정 전경으로, 0.5배만큼 확장한 마스크 바깥을 확정 배경으로 정하되 몸통 영역은 미확정으로 둡니다.

- **FBA: Main:** Main alpha에서 위 규칙으로 트라이맵을 만들어 FBA 실행.
- **FBA: clothing:** 현재 옷 모듈 alpha에서 같은 규칙으로 트라이맵을 만들어 FBA 실행. 기존 결합의 재현입니다.
- **FBA + post:** Main FBA 이후 Main 트라이맵 미확정이고 옷 트라이맵 확정 전경, 옷 gate ≥ 0.95, 옷 alpha ≥ 0.9, Main FBA alpha > 0.75인 픽셀만 두 alpha의 큰 값으로 올립니다. Main의 확정 배경은 유지하고 FBA 전경색은 다시 추정하지 않습니다. 이 높은 alpha와 침식 마스크는 보수적인 **대리 근거**이지 보정된 옷 확률이나 확실한 옷 라벨이 아닙니다.
- **FBA: guarded:** 기존 Guarded FBA처럼 Main 트라이맵의 미확정 중 옷 트라이맵 확정 전경과 Main FBA alpha > 0.75가 겹치는 픽셀만 확정 전경으로 바꿔 FBA를 새로 실행합니다. Main의 확정 배경을 풀지 않습니다.

새 후처리의 기준은 사진별 조정 없이 고정한 실험적 규칙이고, Guarded의 0.75는 이전에 이 사진들을 보며 정한 기준입니다. 예를 들어 shot5에서는 `FBA: clothing`이 빠진 재킷을 되찾지만, Main을 보존하는 post와 guarded는 거의 복원하지 못합니다. test1에서는 옷 트라이맵 결합이 옷과 함께 양옆 배경도 짙게 남기며, test9에서는 여러 FBA 결과에 배경과 그림자가 남습니다. shot2의 post는 재킷 alpha를 올리지만 원래 FBA 전경색을 그대로 재사용하므로 얼룩처럼 보일 수 있습니다. 합성은 FBA의 경우 추정 전경과 alpha를 sRGB에서 흰색에 합성하고 Main·Clothing은 앱의 선형광 합성을 쓰므로 색상 비교는 동일 조건이 아닙니다.

**정답 마스크가 없으므로** CSV의 alpha 평균, 0.5 초과 픽셀과 FBA 대비 변화는 정확도나 옷 복원율이 아닙니다. 새 랜드마크 캐시로 검출했으므로 이전 실행의 숫자와 달라질 수 있습니다. 이 비교는 FBA 가중치를 비상업적 로컬 연구에만 사용했습니다. 모델 준비 후 아래 명령으로 재현합니다.

```bash
uv run --extra fba --group dev python -m tools.eval_fba_combinations --out out/fba-combinations/live
uv run --extra fba --group dev python -m tools.eval_fba_combinations --full-grid --out out/fba-combinations/full-grid
```

## 최신 실험: 12장 전체

`Previous clothing`은 직전 앱 코드 `4276adf6a4cba13d0b98f4a8e333498e8a5c40db`, `Clothing (NumPy)`는 현재 근접 촬영 보정을 포함한 앱 코드입니다. `FBA`는 Main alpha로, `NumPy + FBA`는 현재 옷 alpha로 **같은 규칙으로 만든 자동 트라이맵**을 받아 원본 사진을 처리합니다. SCHP와 Sapiens는 FBA가 아닌 **Main alpha**에 파싱 결과를 결합합니다. `clothing`은 옷 클래스만, `person`은 머리카락을 포함한 전체 인물 예측을 사용합니다. 여기서 Sapiens는 비율 보존 검은 패딩과 고정 감마 0.55를 모델 입력에만 적용합니다.

12장의 동일한 640 px 너비 입력과 랜드마크로 라이브(`s=2`) 및 [전체 격자 `s=1`](docs/evaluation/clothing/research-gamma/full-grid/comparison.jpg)를 평가했습니다. [라이브 CSV](docs/evaluation/clothing/research-gamma/live/measurements.csv)와 [전체 격자 CSV](docs/evaluation/clothing/research-gamma/full-grid/measurements.csv)에 12장 모두 남겼습니다. `shot2`–`shot4`는 비슷한 장면이라 그림에는 대표인 `shot2`만 표시합니다. 아래 관찰의 좌우는 사진을 보는 방향입니다.

| 사진 | NumPy 옷과 배경 | SCHP person | Sapiens 패딩 + 감마 person | Sapiens 대 SCHP 픽셀: 라이브; 전체 격자 |
|---|---|---|---|---|
| shot2 | 재킷은 남지만 경계가 거칠고 전체 격자에서는 조각납니다. | 재킷과 머리가 더 남습니다. | 재킷과 머리가 비슷하게 남지만 경계가 완벽하지 않습니다. | +343/-690; +382/-703 |
| shot3 | 왼쪽 재킷 일부가 빠집니다. | 재킷이 더 남지만 왼쪽 끝이 일부 비칠 수 있습니다. | 재킷과 머리가 남지만 경계가 완벽하지 않습니다. | +444/-644; +453/-641 |
| shot4 | 오른쪽 배경 물체가 남습니다. | 재킷이 더 남고 물체가 줄어듭니다. | 재킷이 남고 물체가 줄지만 경계가 거칩니다. | +446/-629; +447/-630 |
| shot5 | 근접 재킷의 양쪽이 돌아오지만 오른쪽 어깨가 거칩니다. | 재킷과 잘린 머리가 돌아옵니다. | 재킷과 머리가 돌아오지만 경계가 완벽하지 않습니다. | +1,259/-750; +1,259/-750 |
| test1 | 어두운 방 물체와 옷 구멍이 남습니다. | 옆 배경이 줄고 상의가 더 연결됩니다. | 패딩만 쓴 경우보다 상의가 많이 돌아오지만 뚜렷한 구멍과 잘린 머리가 남습니다. | +1,382/-4,094; +1,458/-3,964 |
| test2 | 양옆의 어두운 물체가 남습니다. | 머리와 배경은 개선되지만 오른쪽 어깨·소매가 잘립니다. | 감마를 써도 상의 거의 전체가 사라집니다. | +0/-54,200; +0/-54,202 |
| test3 | 목 옆 패임과 오른쪽 배경이 남습니다. | 배경이 줄고 상의와 머리가 더 남습니다. | 회색 상의는 남지만 머리가 많이 잘립니다. | +626/-10,381; +627/-8,780 |
| test4 | 파란 물체와 어두운 배경이 남습니다. | 옷 주위 물체가 줄고 머리가 더 남습니다. | 상의 거의 전체가 사라집니다. | +0/-56,237; +0/-56,057 |
| test5 | 오른쪽 흰 어깨와 왼쪽 벽 조각이 남습니다. | 벽 조각이 줄지만 옷과 목 경계가 완벽하지 않습니다. | 벽은 줄지만 상의 그림에 뚜렷한 구멍이 납니다. | +6,993/-22,509; +6,792/-22,533 |
| test6 | 어깨 뒤 벽 누출이 남습니다. | 벽은 줄지만 밝은 전경에 흰 구멍이 남습니다. | 벽은 줄지만 상의와 아래 전경이 크게 잘립니다. | +10,591/-41,028; +10,227/-41,082 |
| test8 | 칼라 패임과 각진 경계가 남습니다. | 상의가 더 이어지지만 오른쪽 소매에 잔여물이 보입니다. | 패딩만 쓴 경우보다 상의 일부가 돌아오지만 점무늬처럼 구멍이 납니다. | +4,898/-4,278; +4,871/-4,184 |
| test9 | 후드와 머리가 잘리고 배경이 연결됩니다. | 후드와 머리가 더 남지만 옷 경계가 거칩니다. | 후드와 머리가 더 남지만 그림자 쪽 회색 잔여와 거친 경계가 남습니다. | +13,681/-915; +13,378/-915 |

마지막 열은 전체 이미지에서 alpha 0.5를 기준으로 `+`는 Sapiens person에만, `-`는 SCHP person에만 남은 픽셀 수입니다. 옷뿐 아니라 머리·배경도 포함하며 사진 크기도 달라 정확도나 복원율이 아닙니다. 각 파서와 Main의 비교 및 옷 전용 수치는 CSV에 있습니다.

**실제 사진에는 정답 마스크가 없습니다.** 12장은 독립적인 장면이 아니며 결합 임계값과 감마도 같은 사진을 보며 탐색했습니다. 이전 옷 코드 대비 alpha 0.5 기준 shot5는 라이브에서 추가 31,525·제거 0개, 전체 격자에서 추가 31,292·제거 0개였고 나머지 11장의 이진 마스크는 같았습니다. 추가 픽셀에는 배경도 포함될 수 있습니다. SCHP person은 **이 예시들의 오프라인 비교 기준**으로 더 유용하지만 test2 옷을 자르기도 합니다. Sapiens와 새 HSV 후보도 전체적으로 신뢰할 만하지 않아 앱에 학습 파서나 새 HSV 로직을 추가하지 않았습니다.

### 적용한 방법과 선택

**앱의 근접 촬영 보정.** 기존 가슴 표본이 처리 격자에서 50개 미만이고 턱 아래가 눈–턱 거리 `ec`의 0.3배보다 많이 보일 때, 얼굴 너비 `fw`를 기준으로 화면 아래쪽 양옆의 좁은 띠를 옷 표본 후보로 씁니다. 후보 시작 높이는 `max(chin + 0.25 ec, chin + 0.75 × 턱 아래 화면 높이)`, 좌우 거리는 중심에서 `0.4–0.85 fw`입니다. 어깨 gate는 `chin - 0.45 ec`부터 양옆으로 넓히고 그 주변을 옷용 배경 표본에서 보호합니다. 배경 모델과의 정규화 색 거리가 0.9 이하인 후보는 제외하고, 남은 후보나 배경 후보가 50개 미만이면 추가 판정을 건너뛰고 업데이트 프레임의 옷 warm state도 비웁니다. 실제로 이 조건은 두 모드 모두 12장 중 shot5에서만 작동했습니다. 후보 608개 중 590개(라이브), 2,618개 중 최대 2,000개를 뽑아 1,998개(전체 격자)를 사용했습니다.

이 보정은 얼굴 랜드마크에 기반한 자세 제약과 기존 얼굴 연결성 처리를 활용하며 별도의 포즈 모델이나 사진별 예외를 쓰지 않습니다. 경계 강도에 맞춰 어깨 경로를 자르거나 침식 뒤 얼굴 연결 부분만 남기는 대안도 시험했지만 test5·test6의 흰 상의나 test9의 후드가 잘려 채택하지 않았습니다. 서로 비슷한 벽과 옷은 근접 표본에서도 구분하기 어려울 수 있고, 어깨를 가리거나 비정상적인 자세에서는 표본 가정이 틀릴 수 있습니다.

**오프라인 SCHP 파싱.** [Self-Correction for Human Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)의 LIP 20개 클래스 모델을 변환한 [ONNX 체크포인트](https://huggingface.co/pirocheto/schp-lip-20/tree/7c0028fe34dcd8bd534ffa291a9086cdc856e4a9)를 CPU에서 실행했습니다. 원본 추론의 BGR 채널 순서와 정규화 값을 사용하고, 입력을 검은 정사각형으로 패딩한 뒤 Pillow 선형 보간으로 473 × 473에 맞춥니다. 이는 원본의 affine 변환이나 변환 모델 카드의 RGB 정사각형 늘이기와 픽셀 단위로 동일하지 않습니다. 20개 logits의 softmax에서 옷 클래스 5·6·7·10·11과 배경 0을 사용합니다.

두 결합은 Main alpha의 턱 아래에서 SCHP 배경 확률이 0.65–0.95로 커질수록 기존 alpha를 줄이되 중심 목은 보호합니다. 옷 또는 전체 인물 확률을 0.15–0.85에서 0–1로 변환해 얼굴과 연결된 후보만 더하고 얼굴은 유지합니다. 옷 전용 결과는 얼굴 기반 옷 gate도 적용합니다. 연결성은 축소 격자에서 근사하므로 완벽한 사람 식별이 아닙니다. 이는 논문 자체의 표준 결합법이 아니라 같은 12장을 보며 정한 실험적 휴리스틱입니다. SCHP의 오분류, 특히 어깨 손실이나 옷에 생긴 구멍이 결합 결과에도 전달됩니다.

원본 코드와 변환 모델 카드는 각각 MIT로 표시되어 있지만, 원 학습 데이터와 가중치의 재배포·상업적 사용 권리는 별도로 확인하지 못했습니다. 가중치 두 파일은 약 255 MiB이며 리비전 `7c0028fe34dcd8bd534ffa291a9086cdc856e4a9`, SHA-256 `41c9d6c753a387f794bf3cbd9aed5e20c81a934fdc91731475bd46138937f84c`와 `be931b91c76ed7707b5426abbccee98620a122b777c7cbd1f771a122803666e7`로 고정·검증합니다. 앱의 계획은 추가 학습 분할 모델을 허용하지 않으므로 ONNX Runtime과 Pillow는 `parsing` 선택 의존성으로만 두고 앱에는 연결하지 않았습니다. 한 번씩 실행한 라이브 SCHP 추론·전후처리 중앙값은 약 190 ms였으며 모델 로딩과 결합, 전체 앱 시간은 제외한 참고값입니다.

**오프라인 Sapiens 파싱.** 공식 [Sapiens 0.3B 분할 TorchScript 체크포인트](https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/tree/18e07d7474a00d30728f22de33325741ce27046f)를 실행했습니다. 파일 크기는 1,358,871,599바이트, 리비전은 `18e07d7474a00d30728f22de33325741ce27046f`, SHA-256은 `735a9a8d63fe8f3f6a4ca3d787de07e69b1f9708ad550e09bb33c9854b7eafbc`입니다. [코드](https://github.com/facebookresearch/sapiens/blob/main/LICENSE)와 모델 카드는 CC BY-NC 4.0으로, 비상업 용도만 허용합니다.

입력은 RGB 1024 × 768이며 평균 `[123.5, 116.5, 103.5]`, 표준편차 `[58.5, 57, 57.5]`로 정규화합니다. apparel·하의·상의 확률(클래스 1·12·22)을 합하고 배경은 클래스 0을 씁니다. 두 파서에 위에서 설명한 같은 Main-alpha 결합을 쓰지만 임계값은 SCHP를 보며 정했고 클래스 정의와 확률 보정도 다를 수 있어 모델만 바꾼 공정한 통제 비교는 아닙니다.

12장과 두 격자에서 전체 프레임의 세 입력을 시험했습니다: [공식 데모 방식 늘이기](docs/evaluation/clothing/research-v2/live/comparison.jpg)([전체 격자](docs/evaluation/clothing/research-v2/full-grid/comparison.jpg)), [비율 보존 검은 패딩](docs/evaluation/clothing/research-letterbox/live/comparison.jpg)([전체 격자](docs/evaluation/clothing/research-letterbox/full-grid/comparison.jpg)), 패딩과 고정 감마 0.55(맨 위 결과). 각 디렉터리에 12행 CSV도 있습니다. 늘이기는 shot5와 test1/2/3/4/8의 옷을 대부분 잃었습니다. 패딩은 shot5와 test3을 복원했지만 test1/2/4의 옷을 대부분 지우고 test5/6에 구멍, test8에 얼룩을 남겼습니다. 같은 사진을 보고 추가한 감마는 패딩만 쓴 경우보다 test1/8의 옷 일부를 되찾았지만 test2/4는 여전히 상의 거의 전체가 사라지고 test5/6에도 큰 구멍이 남습니다. 감마는 모델 입력에만 적용하고 합성은 원본 RGB를 사용합니다. 어떤 변형도 앱에 넣기 적합하지 않았습니다.

Apple M4 Pro, RAM 24 GiB, Python 3.12.13, PyTorch 2.15.0, ONNX Runtime 1.32.0에서 한 번 측정한 패딩 라이브 결과의 추론·전후처리 중앙값은 Sapiens MPS 약 1.6초, SCHP CPU 약 0.18초였습니다. 장치가 다르고 모델 로딩·결합·나머지 앱 시간은 제외된 탐색적 수치입니다. 동일 shot5 입력의 실제 CPU와 MPS Sapiens 옷/배경 확률 최대 차이는 0.000023 이하였고 두 확률의 0.5 판정 차이는 없었습니다.

**HSV·기하·연결성 후보.** 별도의 오프라인 NumPy 가설로 HSV 배경 색을 모델링하고 얼굴 기반 중앙 몸통을 보호한 뒤, 배경 표본과 연결된 유사 색 영역을 얼굴 연결성 정리 전에 제거했습니다. 전체 포즈 추정 논문을 그대로 재현한 것은 아닙니다. [라이브](docs/evaluation/clothing/hsv-probe/live/comparison.jpg)와 [전체 격자](docs/evaluation/clothing/hsv-probe/full-grid/comparison.jpg) 비교 및 CSV에 12장 모두 포함했습니다. 현재 NumPy 대비 alpha 0.5 기준 제거된 픽셀 합은 각각 12,889개와 26,669개지만 정답이 없어 옷과 배경을 구분하는 수치가 아닙니다.

`tests/test_hsv_probe.py`의 정답이 있는 합성 반례에서는 벽과 비슷한 소매 2,150픽셀 중 현재 NumPy가 라이브 2,020개·전체 격자 1,997개를 남긴 반면 후보는 692개·679개만 남겼고, 이 장면에서 배경 제거는 **0개**였습니다. 마스크가 작아진 것을 개선으로 간주하지 않고 후보를 기각했습니다. 앱의 기존 근접 보정은 유지하며 새 후보로 앱 코드를 바꾸지 않았습니다.

### 다른 연구 후보

| 후보 | 접근성·제약과 결정 |
|---|---|
| [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) | 고정한 변환 모델을 CPU에서 실행했습니다. 원본 코드와 변환 모델 카드는 MIT로 표시되지만 원 가중치와 학습 데이터 권리는 별도 확인이 필요합니다. 오프라인 비교 전용입니다. |
| [Sapiens](https://github.com/facebookresearch/sapiens) | 공식 0.3B 모델을 로컬에서 실행했으나 코드·가중치가 CC BY-NC 4.0이고 관찰된 실패 때문에 앱에 채택하지 않았습니다. |
| [Sapiens2](https://github.com/facebookresearch/sapiens2) | 공식 0.4B 분할과 1B 매팅 가중치는 공개되어 있지만 각각 약 1.63 GB와 6.16 GB가 추가됩니다. [라이선스](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)는 생체정보 처리 등 일부 용도를 금지하며 사용 시 약관에 동의하게 됩니다. 증명사진 용도가 해당 제한에 속하는지는 법률 검토가 필요합니다. 다운로드·실행하지 않았습니다. |
| [Spectrum](https://arxiv.org/abs/2508.06032) | 착용 의류 파싱 논문이지만 확인 당시 논문과 프로젝트 페이지에서 재현 가능한 코드·가중치 다운로드를 찾지 못했고 모델 사용 조건도 확인하지 못했습니다. 실행하지 않았습니다. |
| [D2FP](https://github.com/cvlab-yongin/D2FP) | 코드는 MIT로 표시되며 [공식 릴리스 가중치](https://github.com/cvlab-yongin/D2FP/releases/tag/weights)는 각각 약 958 MB입니다. 공식 설치에는 Detectron2와 CUDA 확장이 필요하고 Python 디버그 fallback도 그 확장을 import합니다. 이 Mac에는 Detectron2와 CUDA가 없으며 가중치·데이터 권리도 별도 검토가 필요해 실행하지 않았습니다. |
| [SAM 3](https://github.com/facebookresearch/sam3) | 가중치는 [수동 승인 대상](https://huggingface.co/facebook/sam3)이고 별도 [SAM 라이선스](https://github.com/facebookresearch/sam3/blob/main/LICENSE)를 따릅니다. 공식 경로는 CUDA를 요구하고 체크포인트는 약 3.45 GB입니다. 실행하거나 외부 데모에 사진을 보내지 않았으며, 의류 프롬프트도 대상 인물과의 연결 판정이 필요합니다. |
| HSV/morphology와 자세·연결성 아이디어 | NumPy 후보와 기존 얼굴 랜드마크 기반 근접 보정으로 시험했으며 해당 논문들의 전체 시스템 재현은 아닙니다. 새 후보는 위 합성 반례 때문에 기각했습니다. |

실행하지 않은 Sapiens2 [분할 `model.safetensors`](https://huggingface.co/facebook/sapiens2-seg-0.4b/tree/449b3c5335e6722bb94990abdd1aa6e612432f22)의 리비전은 `449b3c5335e6722bb94990abdd1aa6e612432f22`, 크기는 1,626,451,404바이트, LFS SHA-256은 `b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9`입니다. [매팅 `model.safetensors`](https://huggingface.co/facebook/sapiens2-matting-1b/tree/88b79074a58a0e2bb5befc00a237488c43cd94da)는 리비전 `88b79074a58a0e2bb5befc00a237488c43cd94da`, 6,157,412,312바이트, LFS SHA-256 `2a0ea4ea2234e8dbd51e952950d8cedfe3ee2eca9099003784fd2d9cec6bc6e1`입니다. 이는 저장소 메타데이터이며 로컬 다운로드로 검증한 체크섬이 아닙니다.

### 새 비교 재현

FBA와 Sapiens의 비상업 제한을 확인한 뒤 로컬에서 실행합니다. 사진은 전송하지 않고 모델은 Git에서 제외된 `models/`에 저장됩니다. 아래는 Sapiens에 Apple MPS를 쓰며, MPS가 없으면 `--sapiens-device cpu`를 사용합니다.

```bash
uv sync --group dev --extra fba --extra parsing
uv run python tools/fetch_model.py
uv run --extra fba python tools/fba_reference.py --accept-noncommercial
uv run --extra parsing python -m tools.schp_reference --download
uv run --extra fba python -m tools.sapiens_reference --accept-noncommercial
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --out out/clothing-research/stretch/live
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --full-grid --out out/clothing-research/stretch/full-grid
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --out out/clothing-research/pad/live
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --full-grid --out out/clothing-research/pad/full-grid
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --sapiens-gamma 0.55 --out out/clothing-research/gamma/live
uv run --extra fba --extra parsing python -m tools.eval_clothing --fba --schp --sapiens --sapiens-device mps --sapiens-layout letterbox --sapiens-gamma 0.55 --full-grid --out out/clothing-research/gamma/full-grid
uv run python -m tools.eval_hsv_probe --out out/clothing-research/hsv/live
uv run python -m tools.eval_hsv_probe --full-grid --out out/clothing-research/hsv/full-grid
uv run --extra fba --extra parsing pytest -q
```

각 평가 명령은 비교 그림 한 장과 12행 CSV를 만듭니다. HSV 후보에 필요한 랜드마크 캐시가 생성되도록 파서 비교를 먼저 실행합니다. Main과 파서 결합은 Main의 같은 국소 배경 추정과 선형광 합성을 사용하지만, FBA는 추정한 전경과 alpha를 sRGB에서 합성하므로 색상은 완전히 같은 조건이 아닙니다. FBA 자동 트라이맵과 측정의 세부 조건은 아래에 설명합니다. 고정된 Main과 직전 옷 코드의 Git 커밋이 로컬에 있어야 하며, 랜드마크 캐시가 달라지면 결과도 달라질 수 있습니다.

## 이전 옷 개선 및 FBA 실험: 사진 파일 12개

아래 표와 그림의 수치는 **근접 촬영 보정 전** 코드 `4276adf6a4cba13d0b98f4a8e333498e8a5c40db`를 기준으로 한 이전 실행입니다. 예를 들어 아래 shot5의 표본 0개와 FBA 결합 변화 0개는 당시 결과이지 현재 결과가 아닙니다.

`docs/evaluation/test`의 `shot2`–`shot5`, `test1`–`test6`, `test8`, `test9`를 비교했습니다. `docs/evaluation/crown-crop/originals/`의 네 사진은 `shot2`–`shot5`와 바이트 단위로 같아 다시 세지 않았습니다.

12개 파일이 서로 독립적인 12개 장면은 아닙니다. `shot2`–`shot4`는 같은 장면의 비슷한 사진이므로 비교 그림과 트라이맵 그림에는 대표인 `shot2`만 보여 주고, 구도가 다른 `shot5`는 유지했습니다. 표와 CSV에는 12장 모두 남아 있습니다.

기존 기준선은 main의 `e96bb08cb678634207ea3aa45f086152c4c96f11`, 옷 개선 코드는 `06ec619158d68d7c3417fd47ce8fd3703ae269ca`입니다. FBA 기준선은 main alpha와 랜드마크에서 만든 트라이맵을, 기존 결합 결과는 옷 개선 alpha와 같은 랜드마크에서 만든 트라이맵을 사용합니다. 보수적 결합인 Guarded FBA는 아래에서 설명합니다. 모두 옷을 포함한 **너비 640 px 전체 프레임**을 처리하며, 앞선 머리 crop 전용 FBA 결과가 아닙니다. 같은 입력과 랜드마크를 사용하고 relighting strength는 0, 합성 배경은 흰색으로 맞췄습니다. 앱 결과는 각각 새 `Matter`에서 `update=False`로 실행해 이전 사진의 색 모델을 이어받지 않았습니다.

**실제 사진에는 정답 마스크가 없습니다.** 바로 아래 첫 표의 숫자는 main alpha가 0.5 이하이고 개선 후 0.5를 넘은 픽셀 수이며 FBA와의 비교 수치가 아닙니다. 실제 옷 픽셀 수, 정확도, 재현율이 아니며 추가된 배경도 포함합니다. 사진 크기가 달라 서로 다른 사진의 숫자를 품질 점수처럼 비교할 수도 없습니다. 관찰은 라이브 결과 기준이고, 좌우는 사진을 보는 방향입니다.

| 사진 | 개선 후 추가: 라이브 | 개선 후 추가: 전체 격자 | 개선 후 라이브 결과와 남은 문제 | FBA 라이브 관찰 |
|---|---:|---:|---|---|
| shot2 | 15,927 | 5,355 | 카키 재킷이 많이 돌아왔지만 바깥 소매와 작은 밝은 구멍이 남습니다. | 재킷과 머리가 더 넓게 남지만 재킷 일부가 반투명합니다. |
| shot3 | 7,916 | 14,472 | 양쪽 재킷이 더 남지만 왼쪽 소매는 상당 부분 빠집니다. | 재킷이 더 남지만 왼쪽은 일부 투명하고 배경도 조금 남습니다. |
| shot4 | 6,699 | 6,157 | 재킷 바깥이 늘지만 오른쪽 방 물체도 남습니다. | 재킷 대부분이 남지만 오른쪽 배경도 일부 남습니다. |
| shot5 | 0 | 0 | 근접 촬영으로 옷 표본이 없어 추가 복원을 건너뜁니다. | 머리는 더 넓게 남지만 재킷 상당 부분은 여전히 사라집니다. |
| test1 | 26,880 | 10,719 | 검은 무늬 상의가 늘지만 어깨 구멍과 어두운 배경 누출이 남습니다. | 머리와 옷이 더 남지만 방 물체와 배경 안개가 뚜렷합니다. |
| test2 | 15,067 | 16,128 | 무늬 상의가 늘지만 옆의 어두운 물체도 계속 연결됩니다. | 머리와 옷이 남지만 양옆에 어두운 방 물체와 안개가 남습니다. |
| test3 | 7,138 | 7,011 | 목·어깨 주변 회색 상의가 늘지만 목 옆 패임이 남습니다. | 머리와 어깨가 더 온전하지만 오른쪽 배경 안개가 남습니다. |
| test4 | 7,185 | 7,109 | 어깨 주변 상의가 늘지만 파란 물체와 옆의 어두운 배경이 남습니다. | 머리와 어깨가 더 온전하지만 양옆의 어두운 물체도 남습니다. |
| test5 | 12,618 | 12,714 | 오른쪽 흰 어깨가 돌아오지만 왼쪽 벽 조각은 그대로 남습니다. | 상의가 남지만 벽 질감과 왼쪽의 어두운 벽 조각도 남습니다. |
| test6 | 11,753 | 12,569 | 흰 상의 바깥이 늘지만 어깨 뒤 벽도 새로 포함되는 퇴행이 있습니다. | 상의가 남지만 머리 뒤에 희미한 벽 질감이 남습니다. |
| test8 | 15,664 | 14,794 | 검은 소매 바깥이 돌아오지만 칼라 패임과 각진 경계가 남습니다. | 상의가 더 온전하지만 천 배경의 안개가 남습니다. |
| test9 | 34,428 | 15,183 | 어두운 후드가 늘지만 그림자 쪽 누락, 배경 연결과 잘린 머리 경계가 남습니다. | 머리와 후드가 더 넓게 남지만 넓은 배경 안개가 남습니다. |

### FBA와 옷 개선 + FBA: 12장

아래 숫자는 라이브 결과에서 **기존 FBA 대비 결합 결과**의 추가 / 제거 픽셀입니다. 두 alpha를 각각 0.5로 나눴을 때 배경에서 전경으로, 또는 전경에서 배경으로 바뀐 전체 프레임 픽셀을 셉니다. 실제 옷 픽셀이나 정답 오류가 아니며 배경 누출도 포함됩니다.

| 사진 | FBA 대비 추가 / 제거 | 결합 결과 관찰 |
|---|---:|---|
| shot2 | 2,079 / 0 | 반투명하던 재킷이 더 불투명해지지만 오른쪽 배경 일부도 남습니다. |
| shot3 | 1,111 / 999 | 왼쪽 재킷이 덜 비치지만 배경 일부도 남습니다. |
| shot4 | 22 / 433 | 재킷은 대부분 남고 기존 FBA와 비슷하지만 오른쪽 배경 물체도 있습니다. |
| shot5 | 0 / 0 | 옷 표본이 없어 두 트라이맵과 FBA alpha가 같으며 재킷 누락도 그대로입니다. |
| test1 | 18,831 / 439 | 상의와 함께 양옆의 어두운 방 물체가 기존 FBA보다 더 짙게 남습니다. |
| test2 | 8,747 / 1 | 상의 주변이 남지만 양옆의 어두운 배경도 짙게 연결됩니다. |
| test3 | 2,044 / 10 | 어깨는 남지만 오른쪽 배경 안개도 남습니다. |
| test4 | 2,979 / 91 | 머리와 어깨는 남지만 양옆의 어두운 물체와 안개도 남습니다. |
| test5 | 1,090 / 43 | 흰 상의는 남지만 왼쪽 벽 조각과 벽 질감도 남습니다. |
| test6 | 10,256 / 36 | 오른쪽 어깨 뒤 벽이 기존 FBA보다 더 짙게 남습니다. |
| test8 | 38 / 2,640 | 검은 상의와 천 배경의 희미한 안개가 남으며 기존 FBA와 일부 alpha가 달라집니다. |
| test9 | 65,273 / 0 | 후드는 더 진하지만 머리 주변과 왼쪽 위 배경도 훨씬 짙어집니다. |

### Guarded FBA: 12장

기존 옷 개선 결과를 트라이맵으로 바꾸면 새 영역이 확정 전경이 되면서 주변 확정 배경도 미확정으로 풀립니다. 라이브 test6과 test9에서 기존 결합 트라이맵의 새 확정 전경은 각각 16,263개와 31,940개, 확정 배경에서 미확정으로 바뀐 픽셀은 2,885개와 6,788개입니다. 이 두 사진에서 확정 배경이 확정 전경으로 직접 바뀐 픽셀은 없었습니다. 새 확정 전경에는 실제 옷이 많이 포함되지만, 넓어진 미확정 영역에서 FBA가 벽이나 그림자도 남길 수 있습니다. 이 수치는 영역 변화이지 배경 오분류 수는 아닙니다.

Guarded FBA는 **main 트라이맵을 유지**하고, 미확정 픽셀이 옷 개선 트라이맵에서는 확정 전경이며 첫 번째 main 기반 FBA alpha가 0.75를 넘을 때만 확정 전경으로 바꿉니다. main의 확정 배경을 미확정으로 풀지 않고, 원본 사진으로 FBA를 한 번 더 실행합니다. 0.75는 이 사진들을 살펴본 뒤 선택한 실험적 기준으로 독립 데이터에서 검증하지 않았습니다.

다음 숫자는 라이브에서 기존 옷 개선 + FBA 대비 Guarded FBA의 alpha 0.5 기준 추가 / 제거 픽셀입니다. 옷과 배경이 함께 포함되므로 정확도나 옷 복원량이 아닙니다.

| 사진 | 기존 결합 대비 추가 / 제거 | Guarded FBA 관찰 |
|---|---:|---|
| shot2 | 572 / 875 | 재킷은 FBA보다 선명하지만 기존 결합보다 일부 덜 진합니다. |
| shot3 | 397 / 4 | 재킷이 대체로 유지되지만 왼쪽이 일부 비칩니다. |
| shot4 | 51 / 8 | 기존 결합과 비슷하며 오른쪽 배경 물체도 남습니다. |
| shot5 | 0 / 0 | 옷 표본이 없어 기존 FBA와 같고 재킷 누락도 남습니다. |
| test1 | 134 / 11,701 | 상의가 남지만 양옆 방 물체와 안개도 보입니다. |
| test2 | 24 / 1,728 | 상의가 남지만 어두운 방 물체도 연결됩니다. |
| test3 | 6 / 1,895 | 어깨는 남지만 오른쪽 배경 안개도 남습니다. |
| test4 | 5 / 4,016 | 상의와 머리는 남지만 양옆 물체가 남습니다. |
| test5 | 0 / 1,072 | 흰 상의와 함께 왼쪽 벽 조각과 벽 질감이 남습니다. |
| test6 | 0 / 9,456 | 상의가 남고 어깨 뒤 벽은 기존 결합보다 옅지만 완전히 없어지지는 않습니다. |
| test8 | 803 / 3 | 상의가 남지만 배경 안개가 남고 기존 결합과 경계가 다릅니다. |
| test9 | 0 / 48,623 | 후드가 남고 머리 주변 배경은 기존 결합보다 옅지만 여전히 안개가 있습니다. |

라이브 실험에서 색 거리 차이 `clothing_evidence`에 0, 0.5, 1.0 초과 조건을 넣는 대안은 배경이 줄어드는 대신 0.5부터 shot2 재킷이 다시 빠졌습니다. 옷 음영 허용폭을 `(0.6, 1.5)`, `(0.4, 2.5)`, `(0.25, 4.0)`으로 넓히는 대안은 옷을 더 남기기도 했지만 test6과 test8에서 벽까지 더 포함해 둘 다 채택하지 않았습니다. Guarded FBA는 test6·test9의 누출을 줄이는 제한적인 오프라인 실험이지, 옷 개선 단독 결과나 앱 알고리즘의 품질을 고친 것이 아닙니다. 첫 FBA가 잘못 판단한 배경을 확정할 수 있고 낮게 판단한 옷을 놓칠 수도 있습니다. 미확정 영역의 예측은 다시 달라질 수 있어 모든 사진에서 개선된다고 볼 수 없습니다.

### 재킷: shot2–shot5

shot2에서는 기존에 거의 사라졌던 카키 재킷이 더 남지만, 모든 소매가 복원되거나 배경이 완전히 제거된 것은 아닙니다. shot5에서는 얼굴이 화면을 크게 차지해 표본 후보가 0개이므로 기존 결과가 그대로 남습니다.

### 어두운 방: test1–test4

test1과 test2에서는 더 넓은 검은 상의와 함께 방의 어두운 물체도 남습니다. 단순히 마스크가 커졌다는 사실만으로 옷을 정확히 분리했다고 판단할 수 없는 사례입니다.

### 밝은 벽과 다른 자세: test5, test6, test8, test9

test5의 오른쪽 흰 어깨는 전보다 넓어졌지만, test6에서는 어깨 뒤의 밝은 벽까지 새로 따라 들어옵니다. 비슷한 색의 옷과 벽을 구분하는 문제는 해결되지 않았습니다.

### FBA 기준선, 결합 방법과 트라이맵

FBA는 사진만이 아니라 확정 전경·확정 배경·미확정 영역을 구분한 **트라이맵**을 함께 받아 전경, 배경과 alpha를 추정합니다. 기존 FBA에는 main alpha로 만든 트라이맵을, 결합 결과에는 **옷 개선 alpha로 같은 규칙을 적용해 만든 트라이맵**을 넣습니다. 둘 다 원본 사진을 FBA에 입력합니다. 즉 FBA 결과에 옷 모델을 다시 적용하는 후처리가 아니라, 옷 개선을 먼저 실행해 FBA의 입력 조건을 바꾸는 실험입니다.

각 alpha를 0.5로 이진화한 뒤 얼굴 너비의 0.06배만큼 침식한 안쪽과 얼굴 마스크를 확정 전경으로 지정합니다. 이진 마스크를 얼굴 너비의 0.5배만큼 팽창한 바깥은 확정 배경으로 지정하되, 턱 아래 랜드마크로 잡은 넉넉한 몸통 영역 안은 확정 전경을 제외하고 미확정으로 둡니다. 몸통 영역의 반폭은 턱부터 눈–턱 거리만큼 내려가면서 얼굴 너비의 1배에서 3.5배까지 정규화 거리의 제곱근에 따라 넓어집니다. 침식과 팽창에는 정사각형 창을 쓰며 반경은 픽셀 단위로 반올림하고 최소 1 px로 둡니다.

![FBA 입력, 자동 트라이맵과 alpha](docs/evaluation/clothing/fba/trimap.jpg)

두 결과는 각각 **main 또는 옷 개선 마스크와 랜드마크에 조건화된 FBA**이며 독립적인 자동 분할 기준선이나 사람이 수정한 트라이맵이 아닙니다. 배경을 확정 전경으로 잘못 지정하면 지울 수 없고, 옷을 확정 배경으로 지정하면 복원할 수 없습니다. 옷 개선에서 잘못 포함된 배경이 결합 트라이맵의 확정 전경이 되면 FBA도 그 배경을 지우지 못할 수 있습니다. 넓은 몸통 미확정 영역은 기존 마스크 바깥 소매를 FBA가 판단할 기회를 주지만 벽 누출도 허용합니다. main 기반 라이브 트라이맵의 미확정 비율은 31.1–59.7%, 중앙값 42.5%로, 입력의 특성이지 품질 점수가 아닙니다. shot2에서는 결합으로 재킷이 더 불투명해지지만 test6과 test9에서는 배경도 더 진해지므로, 결합이 항상 나은 결과라고 판단할 수 없습니다.

FBA도 앱 비교와 같은 640 px 입력을 쓰며 CPU 스레드 4개, 타일 크기 1024 px로 처리했습니다. 12장 모두 타일 하나에 들어가며, 8의 배수가 되도록 패딩해 입력 픽셀 격자를 유지합니다. 확정 트라이맵을 강제하고 FBA가 추정한 전경과 alpha를 sRGB에서 흰 배경에 합성했으며 추가 후처리는 하지 않았습니다. 앱은 선형광 합성과 자체 spill 제거를 사용하므로 합성 색상까지 완전히 같은 조건은 아닙니다. 기존 라이브 단일 실행에서 FBA 호출 시간은 중앙값 291 ms(234–1,491 ms), 기존 결합 FBA 호출은 292 ms(236–683 ms)였습니다. 기존 결합과 Guarded FBA는 같은 모델의 후속 호출이며, Guarded FBA는 첫 FBA 결과도 필요합니다. 이 수치는 모델 로딩·트라이맵 생성·랜드마크 검출·입력 마스크 생성을 제외하며 공정한 속도 비교나 전체 파이프라인 시간, 실시간 성능을 뜻하지 않습니다.

라이브는 640 px 입력에서 색 판정과 필터 계수를 **절반 격자 `s=2`**로 계산합니다. [전체 격자 비교](docs/evaluation/clothing/full-grid-fba/comparison.jpg)는 **같은 640 px 입력**에 촬영 알고리즘의 `s=1`을 적용하고 해당 main 및 옷 개선 alpha에서 두 트라이맵을 다시 만듭니다. 옷 개선 단독 결과에서 shot2와 shot4의 어깨는 이 경로에서 여전히 조각나고, shot3은 라이브보다 많이 추가됩니다. **원본 해상도 저장 결과를 평가한 것이 아니며**, 두 경로의 경계가 같다는 뜻도 아닙니다.

[전체 격자 FBA 트라이맵과 alpha](docs/evaluation/clothing/full-grid-fba/trimap.jpg), [main과 개선 후의 라이브 alpha 비교](docs/evaluation/clothing/comparison.jpg)도 확인할 수 있습니다.

[라이브 CSV](docs/evaluation/clothing/measurements.csv)와 [전체 격자 CSV](docs/evaluation/clothing/full-grid/measurements.csv)에는 12장 각각의 임계값, 표본 후보 수, 적용 여부와 단일 실행 시간이 있습니다. 전체 합계는 라이브에서 추가 161,275개·제거 9개, 전체 격자에서 추가 122,211개·제거 6개입니다. 제거는 기존 alpha가 0.5를 넘고 개선 후 0.5 이하가 된 픽셀입니다. 이진 마스크에는 정리 **전** 추가만 하지만, 이후 정리와 guided filter를 다시 계산하면 기존 경계도 달라질 수 있습니다.

[FBA 라이브 CSV](docs/evaluation/clothing/fba/measurements.csv)와 [FBA 전체 격자 CSV](docs/evaluation/clothing/full-grid-fba/measurements.csv)에는 두 트라이맵의 영역 크기, 기존 결과와 Guarded FBA의 alpha 0.5 기준 추가·제거 픽셀, 타일 수와 처리 시간도 있습니다. 이 차이는 정답 오류 수나 옷 정확도가 아닙니다.

비교 그림은 rule 7 판정과 관계없이 흰 배경 합성을 보여 줍니다. 실제 앱은 촬영 시 rule 7이 실패하면 원래 벽을 유지할 수 있으므로 그림이 곧 저장 결과라는 뜻은 아닙니다.

## 1. 왜 옷이 빠졌나

기존 gate는 두 역할을 함께 합니다. 점수 단계에서 gate 밖은 사람이 있을 수 없다고 판정하면서 동시에 배경 색 표본 후보로 씁니다. 어깨가 가정보다 넓으면 옷이 잘리는 데 그치지 않고, 그 옷이 배경 모델에도 들어갈 수 있습니다.

앞선 [배경 분리 실패 분석](https://github.com/hanbin5/relight/blob/module-b/matting/README_kr.md)에서는 test5의 오른쪽 흰 어깨 끝에서 배경 표본 약 20개가 뽑혀 흰색 군집이 생겼습니다. shot2에서는 카키 재킷이 회색 롤스크린 군집의 어두운 색과 비슷했고, 랜드마크로 잡은 어깨 바깥 영역의 마스크 포함 비율이 59%였습니다. 이 수치는 **이전 원본 해상도 분석의 영역 관찰**이며 정답 옷 마스크의 재현율이 아닙니다. 위 640 px 비교와 직접 전후 수치로 연결하지 않습니다.

기존 사람 색 표본은 얼굴, 머리, 목과 좁은 가슴을 함께 표현합니다. 가슴 가운데 검은 이너가 많으면 바깥 재킷 색은 충분히 대표되지 않을 수 있습니다. 어두운 방에서는 검은 상의와 방 물체의 색이 겹칩니다. 몸통 prior가 0.7이어도 확정 인물은 아니므로 배경 색 증거와 임계값에 의해 옷이 빠질 수 있습니다.

## 2. 옷 전용 색 모델과 배경 표본 보호

### 실험 이유와 선택

처음에는 전역 어깨 gate를 위로 넓히고 가슴 표본을 확장했습니다. 옷은 더 남았지만 전역 배경·사람 모델과 Otsu 임계값이 함께 달라져 shot3과 test9의 머리 경계에도 변화가 보였습니다. 그래서 머리 색 모델을 다시 정의하는 대신 옷 점수만 따로 계산하는 방법을 선택했습니다. 이 첫 시도의 중간 코드와 그림은 보관하지 않아 독립적인 정량 비교나 재현 가능한 ablation 결과는 아닙니다.

흰 옷의 음영 허용폭도 줄이고 실제 옷 표본에 가까운 색만 추가하도록 제한했습니다. 그래도 test6의 벽 누출은 남았습니다. 기존 몸통 마스크를 새 판정으로 대체하는 대신 **근거가 있는 픽셀만 기존 마스크에 추가**해 새 구멍을 줄였지만, 기존에 잘못 포함된 배경도 그대로 남을 수 있습니다.

### 처리 흐름

1. 얼굴 랜드마크의 얼굴 너비 `fw`, 눈–턱 거리 `ec`, 중심선으로 기존보다 이른 높이에서 넓어지는 옷 탐색 영역을 만듭니다. 턱 아래 넓은 보호 구역의 픽셀은 **옷용 배경 표본 후보에서만** 제외합니다. 보호 구역 전체를 인물로 강제하지는 않습니다.
2. 턱 아래 가슴 쪽에서 옷 색 표본 후보를, 보호 구역 바깥의 기존 배경 후보에서 옷용 배경 표본 후보를 만듭니다. 예를 들어 test5 라이브 실행의 후보는 각각 20,243개와 44,745개였고, 각 모델은 그중 최대 2,000개를 사용합니다. 후보 영역에는 실제 옷이나 배경이 아닌 픽셀이 섞일 수 있습니다.
3. 옷용 전경·배경 색 모델을 별도로 만들고, 옷 탐색 영역에서 색 거리와 몸통 prior를 결합합니다. 옷 점수가 기존 Otsu 임계값을 넘고 옷 모델과의 거리가 2.5 미만인 픽셀만 추가합니다.
4. 추가 판정을 기존 이진 마스크에 합친 뒤 기존 opening/closing, 얼굴 연결, 구멍 채우기와 guided filter를 실행합니다. 기존 머리 색 모델과 원래 점수의 정의는 바꾸지 않았지만, 뒤의 연결·경계 처리 결과까지 픽셀 단위로 같다는 보장은 없습니다.

![shot2와 test5의 배경·옷 표본 후보](docs/evaluation/clothing/sample-regions.jpg)

test5에서 기존 배경 후보 49,352개 중 4,607개가 보호 구역 때문에 옷용 배경 후보에서 빠져 44,745개가 남습니다. 파란 영역에는 벽도 포함되며, 모두 오염된 표본이거나 모두 옷이라는 뜻은 아닙니다. 초록 영역도 옷이라고 보장된 영역은 아닙니다.

앱의 영상 처리는 NumPy로 구현했습니다. 별도의 분할·깊이 모델, 보정 프레임, 사진별 수동 영역이나 사진 이름별 임계값은 사용하지 않습니다. 얼굴 랜드마크 검출에는 기존 MediaPipe 모델을 사용합니다.

### 설정과 표본이 부족할 때

| 항목 | 설정 |
|---|---|
| 옷 탐색 영역 | 턱 + 0.2 `ec`부터 시작. 0.8 `ec`에 걸쳐 반폭이 0.6 `fw`에서 2.8 `fw`로 제곱근 형태로 증가하고 바깥 0.25 `fw`에서 감쇠 |
| 배경 표본 보호 | 턱 아래에서 위 반폭 + 0.6 `fw` 안의 기존 배경 후보를 옷용 배경 후보에서 제외 |
| 옷 표본 후보 | 턱 + 0.65 `ec`부터 + 2.5 `ec`까지. 반폭은 1 `ec`에 걸쳐 0.35 `fw`에서 1.0 `fw`로 제곱근 형태로 증가 |
| 목 보호 | 턱 + 0.65 `ec`보다 아래이거나 중심선에서 0.45 `fw` 이상 떨어진 곳만 추가 판정 |
| 옷 prior | 가장자리 0.5, 몸통 안쪽에서 최대 기존 `PRIOR_TORSO=0.7` |
| 색 모델 | 배경 8군집·옷 5군집, 각각 최대 2,000개 표본. 표본 RNG seed는 배경 3·옷 4 |
| k-means | 초기 8회, 이전 모델이 있는 업데이트 프레임은 3회 |
| 음영과 색 근거 | 옷 색의 선형 밝기 배율 0.8–1.2; 색 거리 계산에는 제곱근을 적용. 옷 모델까지의 거리는 기존 상한 2.5 미만이어야 함 |
| 임계값 | 기존 점수의 Otsu 값, 0.35–0.65로 제한 |

기존 옷 후보가 50개 미만이면 위의 근접 촬영 보정을 먼저 시도합니다. 그 후 사용 가능한 옷 표본이나 보호 후 배경 후보가 처리 격자에서 50개 미만이면 이 추가 판정을 건너뜁니다. 업데이트 프레임이면 이전 옷 모델도 비우며, `Matter.reset()`은 기존 모델과 옷 모델을 모두 초기화합니다. 적용 여부와 후보 수는 `ctx.stats["matting"]`에 기록합니다.

**프레임 테두리 fallback의 범위에 주의해야 합니다.** 기존 배경 또는 기존 전경 후보가 50개 미만이면 원래 코드가 배경 후보를 프레임 테두리로 바꿉니다. 이 경우 옷용 배경 후보도 그 테두리에서 보호 구역을 뺀 것이므로, 50개 이상 남으면 테두리 픽셀로 옷용 배경 모델을 학습할 수 있습니다. 표본이 반드시 깨끗한 벽이라는 보장은 없습니다.

보호된 배경 모델은 옷 점수에만 쓰입니다. 기존 머리·배경 모델, 벽 통계, 단색 벽 판정, 국소 spill 추정은 기존 정의를 유지합니다. 따라서 모든 배경 추정이 정화되었다는 뜻은 아닙니다.

## 3. 정답이 있는 합성 회귀 실험

실제 사진과 달리, 320 × 280 합성 장면의 사람 영역은 생성할 때 알고 있습니다. 더 넓고 위에서 시작하는 어깨를 만든 뒤 같은 기준선과 개선 코드를 비교했습니다. 아래 유지율은 **해당 합성 평가 영역** 중 alpha가 0.5를 넘은 비율이며 전체 사진 IoU가 아닙니다.

| 합성 평가 영역 | 평가 픽셀 | 기존 유지율: 라이브 / 전체 격자 | 개선 후: 라이브 / 전체 격자 |
|---|---:|---:|---:|
| 검은 이너 바깥 카키 재킷 | 13,932 | 0% / 0% | 100% / 100% |
| 기존 gate가 0.05 미만인 흰 소매 | 658 | 0% / 0% | 100% / 100% |
| 분리된 어두운 방 물체가 있는 검은 상의 | 26,304 | 95.38% / 95.15% | 100% / 100% |

개선 후 잘못 포함된 전경은 세 장면에서 **전체 이미지**의 약 0.013–0.016%였습니다. 배경 픽셀만을 분모로 한 비율이 아닙니다. 테스트는 평가 영역 유지율 98% 초과와 전체 이미지 중 잘못 포함된 전경 0.5% 미만을 요구합니다. 표본 보호와 가슴이 잘린 프레임에서의 warm-state 초기화도 검사합니다.

이 단순한 장면들은 옷 개선 단독의 의도한 회귀를 확인할 뿐이며 FBA 결합은 평가하지 않았습니다. 실제 사진의 정확도, 같은 색의 옷과 벽, 다양한 자세나 머리 경계의 품질을 입증하지는 않습니다.

## 4. 재현 방법과 한계

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요합니다. 아래 평가는 추적 중인 그림과 CSV를 덮어쓰지 않도록 `out/`에 출력합니다.

```bash
uv sync --group dev
uv run python tools/fetch_model.py

uv run python tools/eval_clothing.py --sample-figure --out out/clothing-report/live
uv run python tools/eval_clothing.py --full-grid --out out/clothing-report/full-grid

uv run pytest tests/test_matting_checker.py -q
uv run pytest -q
```

각 평가 명령은 비교 그림 한 장과 `measurements.csv`를 만듭니다. 그림에는 10장을 보여 주고 CSV에는 12장 모두 남깁니다. `--sample-figure`는 위의 `sample-regions.jpg`도 생성합니다. 기본 기준선은 위의 고정 main 커밋이며 `--baseline SHA`로 바꿀 수 있습니다. 도구는 Git에서 해당 버전의 `matting.py`를 읽고 현재 공통 설정을 사용하므로, 기준선 커밋이 로컬 Git에 있어야 합니다. Main 기준선은 이 브랜치에 추가된 옷 음영·근접 표본 거리 상수를 사용하지 않습니다. 이 명령을 현재 코드로 다시 실행하면 위의 과거 FBA 결합 수치와 shot5 결과가 달라질 수 있습니다.

FBA 비교를 재현하기 전 [저자의 라이선스와 모델 안내](https://github.com/MarcoForte/FBA_Matting)를 확인하세요. 소스는 MIT 라이선스지만 공개 가중치는 **비상업적 용도로만** 사용할 수 있으며, 이 비교가 상업적 사용이나 제품 탑재를 허용하는 것은 아닙니다. 이 제한을 확인한 뒤:

```bash
uv sync --extra fba --group dev
uv run --extra fba python tools/fba_reference.py --accept-noncommercial
uv run --extra fba python tools/eval_clothing.py --fba --out out/clothing-report/live-fba
uv run --extra fba python tools/eval_clothing.py --fba --full-grid --out out/clothing-report/full-grid-fba
uv run --extra fba pytest tests/test_matting_checker.py -q
uv run --extra fba pytest -q
```

명령은 저자의 소스와 약 139 MB 가중치를 Git에서 제외된 `models/fba/`에 내려받으며 사진은 로컬에 남습니다. 어댑터는 소스 리비전 `8dd100d760186150710510ee8d78ed916cdb17a4`와 가중치 SHA-256 `f3f8a2b6cefd7200368b6008a712468ed1debc6658693db31b0f719f658a2967`을 검증하고 `weights_only=True`로 가중치를 읽습니다. FBA 실행은 Guarded FBA를 포함한 비교 그림 한 장, 트라이맵·alpha 비교 그림 한 장과 CSV를 출력합니다. 자료나 트라이맵이 없거나 잘못되면 다른 방법으로 조용히 대체하지 않고 오류를 냅니다.

랜드마크는 이미지 모드로 검출해 기본적으로 `out/clothing-landmarks/`에 캐시하고, 각 사진의 전후 비교에 같은 값을 씁니다. 캐시는 Git에 포함되지 않습니다. 새 환경이나 새 `--cache` 디렉터리에서 다시 검출하면 랜드마크와 수치가 달라질 수 있습니다. 사진이나 얼굴을 찾지 못하면 조용히 제외하지 않고 오류로 종료합니다. `test9.jpg`는 `module-b/person-prior`의 `e8e798c8eee3118bebba4ba8f34c37808f84c250`에서 원본 그대로 가져왔습니다.

이전 FBA 실험의 검증에서는 전체 215개, matting/checker 45개 테스트가 통과했습니다. main과 개선 후의 CSV 시간은 매팅 단계의 단일 초기 실행값으로, 이미지 로딩·랜드마크·relighting을 포함하지 않는 진단값입니다. 지연시간 보장이나 10 fps 달성을 뜻하지 않습니다.

이 변경은 **일반적인 인물 분리 해결책이 아닌 제한적인 옷 복원**입니다. 근접 표본도 충분히 확보하지 못하면 복원을 건너뛰며, test1/test2/test9처럼 색이 비슷한 물체나 test6처럼 옷에 가까운 벽은 남을 수 있습니다. 팔을 꼬거나 가슴이 가려진 경우에는 표본과 어깨 위치 가정도 틀릴 수 있습니다. 원본 해상도 저장, 실제 연속 프레임의 warm start와 랜드마크 흔들림, 지속 웹캠 성능은 평가하지 않았습니다. 수동 정답 마스크와 시간축 평가 전에는 정확도나 실사용 준비 완료를 주장할 수 없습니다.

앱의 일반 사용법은 [main의 README](https://github.com/hanbin5/relight/blob/main/README.md)를 참고하세요. 이 앱은 편집된 사진을 만들므로 여권 등 정부 신분증에는 사용할 수 없습니다.

const videoInput = document.getElementById("videoInput");
const fileName = document.getElementById("fileName");
const videoPreview = document.getElementById("videoPreview");
const analyzeButton = document.getElementById("analyzeButton");
const xGrid = document.getElementById("xGrid");
const xValueText = document.getElementById("xValue");
const progressCard = document.getElementById("progressCard");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const resultCard = document.getElementById("resultCard");
const resultMessage = document.getElementById("resultMessage");
const tip = document.getElementById("tip");

let selectedX = 5;

for (let x = 1; x <= 10; x++) {
  const button = document.createElement("button");
  button.className = "x-button" + (x === selectedX ? " selected" : "");
  button.textContent = `X = ${x}`;
  button.type = "button";
  button.addEventListener("click", () => {
    selectedX = x;
    xValueText.textContent = x;
    document.querySelectorAll(".x-button").forEach(b => b.classList.remove("selected"));
    button.classList.add("selected");
  });
  xGrid.appendChild(button);
}

videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  if (!file) return;

  fileName.textContent = file.name;
  videoPreview.src = URL.createObjectURL(file);
  videoPreview.hidden = false;
  analyzeButton.disabled = false;
});

const tips = [
  "신체 전체와 분석하려는 관절이 화면 안에 들어오도록 촬영하세요.",
  "관절이 가려지거나 화면 밖으로 나가면 포즈 추적 오차가 증가할 수 있습니다.",
  "빠른 동작은 낮은 X값을 사용하는 것이 도움이 될 수 있습니다.",
  "여러 사람이 겹쳐 있으면 분석 대상 추적이 어려워질 수 있습니다."
];

let tipIndex = 0;
function rotateTip() {
  tip.textContent = tips[tipIndex];
  tipIndex = (tipIndex + 1) % tips.length;
}
rotateTip();

analyzeButton.addEventListener("click", async () => {
  const file = videoInput.files[0];
  if (!file) return;

  analyzeButton.disabled = true;
  progressCard.hidden = false;
  resultCard.hidden = true;
  progressBar.style.width = "10%";
  progressText.textContent = "영상 업로드 준비 중...";
  rotateTip();

  const formData = new FormData();
  formData.append("video", file);
  formData.append("x", String(selectedX));

  try {
    progressBar.style.width = "35%";
    progressText.textContent = "서버로 영상 업로드 중...";

    const response = await fetch("/analyze", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "분석 요청에 실패했습니다.");
    }

    progressBar.style.width = "100%";
    progressText.textContent = "업로드 테스트 완료";

    resultCard.hidden = false;
    resultMessage.textContent = data.message;
  } catch (error) {
    progressBar.style.width = "0%";
    progressText.textContent = "오류가 발생했습니다.";
    resultCard.hidden = false;
    resultMessage.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
  }
});

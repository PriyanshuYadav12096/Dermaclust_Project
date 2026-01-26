/* ================= QUIZ ================= */

const questions = [
  {
    id: "skinType",
    question: "How would you describe your skin type?",
    options: ["Oily", "Dry", "Combination", "Normal"]
  },
  {
    id: "acneFrequency",
    question: "Do you experience acne frequently?",
    options: ["Yes", "Sometimes", "Rarely", "Never"]
  },
  {
    id: "sensitivity",
    question: "How sensitive is your skin?",
    options: ["Very sensitive", "Moderately sensitive", "Not sensitive"]
  },
  {
    id: "mainConcern",
    question: "What is your main skin concern?",
    options: ["Acne", "Pigmentation", "Wrinkles", "Dullness"]
  }
];

const userSkinProfile = {
  quizAnswers: {},
  imageProvided: false,
  imageType: null,
  detectedConcerns: []
};

let currentQuestion = 0;
let uploadedImageBase64 = null;

/* ================= SCREEN NAV ================= */

function showScreen(id) {
  document.querySelectorAll(".screen").forEach(s =>
    s.classList.remove("active")
  );
  document.getElementById(id).classList.add("active");
}

/* ================= QUIZ LOGIC ================= */

function goToQuiz() {
  showScreen("quizScreen");
  loadQuestion();
}

function loadQuestion() {
  const q = questions[currentQuestion];
  document.getElementById("questionText").innerText = q.question;

  const optionsDiv = document.getElementById("optionsContainer");
  optionsDiv.innerHTML = "";

  q.options.forEach(option => {
    const div = document.createElement("div");
    div.className = "option";
    div.innerText = option;

    if (userSkinProfile.quizAnswers[q.id] === option) {
      div.classList.add("selected");
    }

    div.onclick = () => selectOption(q.id, option, div);
    optionsDiv.appendChild(div);
  });

  document.getElementById("progressBar").style.width =
    ((currentQuestion + 1) / questions.length) * 100 + "%";
}

function selectOption(questionId, option, element) {
  userSkinProfile.quizAnswers[questionId] = option;
  document.querySelectorAll(".option").forEach(o => o.classList.remove("selected"));
  element.classList.add("selected");
}

function nextQuestion() {
  const q = questions[currentQuestion];
  if (!userSkinProfile.quizAnswers[q.id]) {
    alert("Please select an option");
    return;
  }

  if (currentQuestion < questions.length - 1) {
    currentQuestion++;
    loadQuestion();
  } else {
    showScreen("choiceScreen");
  }
}

function prevQuestion() {
  if (currentQuestion > 0) {
    currentQuestion--;
    loadQuestion();
  }
}

/* ================= IMAGE CHOICE ================= */

function chooseUpload() {
  userSkinProfile.imageType = "upload";
  showScreen("imageInputScreen");

  resetImageUI();

  document.getElementById("chooseImageBtn").style.display = "block";
}

async function chooseScan() {
  userSkinProfile.imageType = "scan";
  showScreen("imageInputScreen");

  resetImageUI();

  const video = document.getElementById("camera");
  const loader = document.getElementById("cameraLoader");

  loader.style.display = "flex";
  video.style.display = "block";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" }
    });

    video.srcObject = stream;

    video.onloadedmetadata = () => {
      loader.style.display = "none";
      document.getElementById("captureImageBtn").style.display = "block";
    };
  } catch (err) {
    loader.style.display = "none";
    alert("Camera access denied");
  }
}


/* ================= IMAGE INPUT ================= */

function resetImageUI() {
  document.getElementById("chooseImageBtn").style.display = "none";
  document.getElementById("captureImageBtn").style.display = "none";
  document.getElementById("retakeBtn").style.display = "none";
  document.getElementById("analyzeBtn").style.display = "none";

  document.getElementById("camera").style.display = "none";
  document.getElementById("previewImage").style.display = "none";
}

function triggerUpload() {
  document.getElementById("imageUpload").click();
}

document.getElementById("imageUpload").addEventListener("change", function () {
  const file = this.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById("previewImage");
    img.src = e.target.result;
    img.style.display = "block";

    uploadedImageBase64 = e.target.result;
    document.getElementById("analyzeBtn").style.display = "block";
  };
  reader.readAsDataURL(file);
});

/* ================= CAMERA ================= */

function captureFromCamera() {
  const video = document.getElementById("camera");
  const canvas = document.getElementById("snapshot");
  const img = document.getElementById("previewImage");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const imageData = canvas.toDataURL("image/png");

  const stream = video.srcObject;
  stream.getTracks().forEach(t => t.stop());

  video.style.display = "none";
  img.src = imageData;
  img.style.display = "block";

  uploadedImageBase64 = imageData;

  document.getElementById("captureImageBtn").style.display = "none";
  document.getElementById("retakeBtn").style.display = "block";
  document.getElementById("analyzeBtn").style.display = "block";
}

function retakePhoto() {
  if (userSkinProfile.imageType === "scan") {
    chooseScan();
  } else {
    resetImageUI();
    document.getElementById("chooseImageBtn").style.display = "block";
  }
}

/* ================= ANALYSIS ================= */

function analyzeImage() {
  document.getElementById("analysisImage").src =
    document.getElementById("previewImage").src;

  userSkinProfile.detectedConcerns = ["Acne", "Pigmentation"];
  showScreen("analysisScreen");
}

function goToRecommendations() {
  showScreen("recommendationScreen");
}

function skipForNow() {
  userSkinProfile.imageProvided = false;
  userSkinProfile.imageType = null;

  // Directly go to recommendations
  showScreen("recommendationScreen");
}

/* ================= SUMMARY ================= */

function generateSummary() {
  showScreen("summaryScreen");
  populateSummary();
}

function populateSummary() {
  const profile = document.getElementById("summaryProfile");
  const concerns = document.getElementById("summaryConcerns");
  const products = document.getElementById("summaryProducts");

  profile.innerHTML = "";
  concerns.innerHTML = "";
  products.innerHTML = "";

  Object.entries(userSkinProfile.quizAnswers).forEach(([k, v]) => {
    const li = document.createElement("li");
    li.innerText = `${k}: ${v}`;
    profile.appendChild(li);
  });

  userSkinProfile.detectedConcerns.forEach(c => {
    const li = document.createElement("li");
    li.innerText = c;
    concerns.appendChild(li);
  });
}

function downloadSummary() {
  const content = document.querySelector(".summary-box").innerHTML;
  const blob = new Blob([content], { type: "text/html" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "DermaClust_Skin_Summary.html";
  a.click();
  URL.revokeObjectURL(url);
}

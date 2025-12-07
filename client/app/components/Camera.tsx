"use client";
import { useRef, useEffect, useState } from "react";
import Image from "next/image";
import PastUploads from "./PastUploads";

interface ProgressData {
  status: string;
  step: number;
  progress: number;
  message: string;
  duration?: string;
  error?: string;
  stats?: AudioStats;
}

interface AudioStats {
  key: string;
  tempo: number;
  tempo_stability: string;
  pitch_tendency: string;
  pitch_offset: number;
}

const Camera = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [view, setView] = useState<"camera" | "uploads">("camera");
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [voiceLevel, setVoiceLevel] = useState(3);
  const [beatsLevel, setBeatsLevel] = useState(3);
  const [noiseLevel, setNoiseLevel] = useState(3);
  const [musicType, setMusicType] = useState<"rap" | "chill" | "crazy">(
    "chill"
  );
  const [processedAudioUrl, setProcessedAudioUrl] = useState<string | null>(
    null
  );
  const [audioStats, setAudioStats] = useState<AudioStats | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processingMessage, setProcessingMessage] = useState("");

  const voiceLevelRef = useRef(voiceLevel);
  const beatsLevelRef = useRef(beatsLevel);
  const noiseLevelRef = useRef(noiseLevel);
  const musicTypeRef = useRef(musicType);

  useEffect(() => {
    voiceLevelRef.current = voiceLevel;
  }, [voiceLevel]);
  useEffect(() => {
    beatsLevelRef.current = beatsLevel;
  }, [beatsLevel]);
  useEffect(() => {
    noiseLevelRef.current = noiseLevel;
  }, [noiseLevel]);
  useEffect(() => {
    musicTypeRef.current = musicType;
  }, [musicType]);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };

    startCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  const startRecording = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, "recording.webm");

        formData.append("musicType", musicTypeRef.current);
        formData.append("beatsLevel", beatsLevelRef.current.toString());
        formData.append("voiceLevel", voiceLevelRef.current.toString());
        formData.append("noiseLevel", noiseLevelRef.current.toString());

        setIsProcessing(true);
        setProgress(0);
        setAudioStats(null);
        setProcessingMessage("Uploading audio...");

        try {
          const response = await fetch("http://localhost:5000", {
            method: "POST",
            body: formData,
          });

          if (response.ok) {
            const data = await response.json();
            const jobId = data.job_id;

            const eventSource = new EventSource(
              `http://localhost:5000/progress/${jobId}`
            );

            eventSource.onmessage = async (event) => {
              try {
                const progressData: ProgressData = JSON.parse(event.data);
                console.log("Progress update:", progressData);

                setProgress(progressData.progress);
                setProcessingMessage(progressData.message);

                if (progressData.status === "complete") {
                  eventSource.close();
                  console.log("Processing complete, downloading audio...");

                  if (progressData.stats) {
                    setAudioStats(progressData.stats);
                  }

                  try {
                    const audioResponse = await fetch(
                      `http://localhost:5000/download/${jobId}`
                    );
                    console.log("Download response:", audioResponse.status);

                    if (audioResponse.ok) {
                      const audioBlob = await audioResponse.blob();
                      console.log("Audio blob size:", audioBlob.size);
                      const audioUrl = URL.createObjectURL(audioBlob);

                      if (processedAudioUrl) {
                        URL.revokeObjectURL(processedAudioUrl);
                      }

                      setProcessedAudioUrl(audioUrl);
                      console.log("Audio URL set:", audioUrl);
                    } else {
                      console.error(
                        "Download failed:",
                        audioResponse.status,
                        await audioResponse.text()
                      );
                    }
                  } catch (downloadError) {
                    console.error("Download error:", downloadError);
                  }

                  setIsProcessing(false);
                  setProgress(0);
                  setProcessingMessage("");
                } else if (progressData.status === "error") {
                  eventSource.close();
                  console.error("Processing error:", progressData.error);
                  setIsProcessing(false);
                  setProgress(0);
                  setProcessingMessage("");
                  setAudioStats(null);
                }
              } catch (parseError) {
                console.error(
                  "Failed to parse progress data:",
                  parseError,
                  event.data
                );
              }
            };

            eventSource.onerror = (err) => {
              console.error("EventSource error:", err);
              eventSource.close();
              setIsProcessing(false);
              setProgress(0);
              setProcessingMessage("");
              setAudioStats(null);
            };
          } else {
            console.error("Server error:", response.statusText);
            setIsProcessing(false);
            setAudioStats(null);
          }
        } catch (error) {
          console.error("Error sending data:", error);
          setIsProcessing(false);
          setProgress(0);
          setProcessingMessage("");
          setAudioStats(null);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  useEffect(() => {
    return () => {
      if (processedAudioUrl) {
        URL.revokeObjectURL(processedAudioUrl);
      }
    };
  }, [processedAudioUrl]);

  return (
    <div className="min-h-screen w-full bg-[#F4F4F0] text-[#111] font-sans flex flex-col relative overflow-hidden selection:bg-[#F45B69] selection:text-white">
      <header className="p-4 md:p-6 flex justify-between items-center border-b border-[#111]/10 bg-[#F4F4F0] z-50 relative">
        <div className="flex items-center">
          <div className="relative h-10 w-40">
            <Image
              src="/image.png"
              alt="System 14"
              fill
              className="object-contain object-left"
              priority
            />
          </div>
        </div>

        <button
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="z-50 p-2 hover:bg-black/5 rounded-full transition-colors"
        >
          <div className="w-6 h-5 flex flex-col justify-between items-end">
            <span
              className={`h-0.5 bg-[#111] transition-all duration-300 ${
                isMenuOpen ? "w-6 -rotate-45 translate-y-2" : "w-6"
              }`}
            ></span>
            <span
              className={`h-0.5 bg-[#111] transition-all duration-300 ${
                isMenuOpen ? "opacity-0" : "w-4"
              }`}
            ></span>
            <span
              className={`h-0.5 bg-[#111] transition-all duration-300 ${
                isMenuOpen ? "w-6 rotate-45 -translate-y-2.5" : "w-2"
              }`}
            ></span>
          </div>
        </button>

        <div
          className={`fixed inset-0 bg-[#F4F4F0] z-40 transition-transform duration-500 ease-in-out flex flex-col items-center justify-center ${
            isMenuOpen ? "translate-x-0" : "translate-x-full"
          }`}
        >
          <div className="flex flex-col gap-8 text-center">
            <h2 className="text-xs font-bold uppercase tracking-widest opacity-50 mb-4">
              Menu
            </h2>
            <button
              onClick={() => {
                setView("uploads");
                setIsMenuOpen(false);
              }}
              className="text-3xl font-bold hover:text-[#F45B69] transition-colors"
            >
              Past Uploads
            </button>
            <button
              onClick={() => {
                setView("camera");
                setIsMenuOpen(false);
              }}
              className="text-3xl font-bold hover:text-[#F45B69] transition-colors"
            >
              Log Out
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        {view === "camera" ? (
          <>
            <div className="flex-1 p-4 md:p-8 flex flex-col justify-center items-center relative bg-[#EAE8E0] lg:border-r border-[#111]/10 lg:min-h-0">
              <div className="relative w-full max-w-2xl aspect-[4/3] md:aspect-video bg-[#111] rounded-2xl overflow-hidden shadow-2xl ring-1 ring-black/5 group transition-all duration-500 hover:shadow-3xl">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover opacity-90 transition-opacity duration-500 group-hover:opacity-100"
                />

                {isRecording && (
                  <div className="absolute inset-0 flex items-center justify-center gap-1.5 bg-black/40 backdrop-blur-sm transition-all duration-300">
                    {[...Array(12)].map((_, i) => (
                      <div
                        key={i}
                        className="w-1.5 md:w-2 bg-[#F45B69] rounded-full animate-wave"
                        style={{
                          height: "20%",
                          animationDelay: `${i * 0.05}s`,
                          animationDuration: "0.8s",
                        }}
                      ></div>
                    ))}
                  </div>
                )}

                <div className="absolute top-6 left-6 flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      isRecording ? "bg-[#F45B69] animate-ping" : "bg-white/50"
                    }`}
                  ></div>
                  <span className="text-white/80 text-[10px] font-medium tracking-widest uppercase">
                    {isRecording ? "Recording" : "Standby"}
                  </span>
                </div>

                <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end">
                  <div className="text-white/90 text-xs font-medium tracking-wider bg-black/20 backdrop-blur-md px-3 py-1 rounded-full border border-white/10">
                    {musicType.toUpperCase()}
                  </div>
                  <div className="flex gap-1">
                    {[1, 2, 3].map((i) => (
                      <div
                        key={i}
                        className="w-1 h-1 bg-white/40 rounded-full"
                      ></div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="flex-1 p-6 md:p-12 bg-[#F4F4F0] flex flex-col gap-8 md:gap-12 justify-center items-center relative">
              <div className="absolute top-10 right-10 w-20 h-20 border-t-2 border-r-2 border-[#111]/5 rounded-tr-3xl pointer-events-none"></div>
              <div className="absolute bottom-10 left-10 w-20 h-20 border-b-2 border-l-2 border-[#111]/5 rounded-bl-3xl pointer-events-none"></div>

              <div className="w-full max-w-md space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-bold uppercase tracking-widest text-[#111]/40">
                    Tuning Mode
                  </label>
                  <div className="h-px flex-1 bg-[#111]/10 ml-4"></div>
                </div>
                <div className="flex gap-2 p-1 bg-[#EAE8E0] rounded-xl">
                  {["rap", "chill", "ionic"].map((type) => (
                    <button
                      key={type}
                      onClick={() =>
                        setMusicType(type as "rap" | "chill" | "crazy")
                      }
                      className={`flex-1 py-3 rounded-lg text-xs font-bold uppercase tracking-wider transition-all duration-300 ${
                        musicType === type
                          ? "bg-white text-[#111] shadow-sm scale-100"
                          : "text-[#111]/40 hover:text-[#111]/70 hover:bg-white/50 scale-95"
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 md:gap-12 w-full max-w-md">
                <MinimalSlider
                  label="Voice"
                  value={voiceLevel}
                  onChange={setVoiceLevel}
                />
                <MinimalSlider
                  label="Beats"
                  value={beatsLevel}
                  onChange={setBeatsLevel}
                />
                <MinimalSlider
                  label="Noise"
                  value={noiseLevel}
                  onChange={setNoiseLevel}
                />
              </div>

              <div className="mt-4 md:mt-8 relative group">
                <div
                  className={`absolute inset-0 bg-[#F45B69] rounded-full blur-xl opacity-20 transition-opacity duration-500 ${
                    isRecording || isProcessing
                      ? "opacity-40 scale-150"
                      : "group-hover:opacity-30"
                  }`}
                ></div>
                <button
                  onClick={toggleRecording}
                  disabled={isProcessing}
                  className={`relative w-20 h-20 md:w-24 md:h-24 rounded-full border-[6px] transition-all duration-500 flex items-center justify-center z-10 ${
                    isProcessing
                      ? "border-[#111]/20 bg-[#EAE8E0] cursor-wait"
                      : isRecording
                      ? "border-[#F45B69] bg-white rotate-180"
                      : "border-[#EAE8E0] bg-[#F45B69] hover:scale-105 hover:shadow-lg"
                  }`}
                >
                  {isProcessing ? (
                    <div className="w-6 h-6 border-2 border-[#111]/20 border-t-[#F45B69] rounded-full animate-spin"></div>
                  ) : (
                    <div
                      className={`transition-all duration-500 ${
                        isRecording
                          ? "w-8 h-8 bg-[#F45B69] rounded-sm"
                          : "w-3 h-3 bg-white rounded-full"
                      }`}
                    ></div>
                  )}
                </button>
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-[10px] font-bold uppercase tracking-widest text-[#111]/30 whitespace-nowrap">
                  {isProcessing
                    ? "Processing..."
                    : isRecording
                    ? "Tap to Stop"
                    : "Tap to Record"}
                </div>
              </div>

              {isProcessing && (
                <div className="mt-12 w-full max-w-md animate-in fade-in slide-in-from-bottom-4 duration-300">
                  <div className="flex justify-between items-center mb-3">
                    <label className="text-xs font-bold uppercase tracking-widest text-[#111]/40">
                      Processing
                    </label>
                    <span className="text-xs font-bold text-[#F45B69]">
                      {progress}%
                    </span>
                  </div>
                  <div className="bg-[#EAE8E0] rounded-2xl p-5">
                    <div className="w-full h-2 bg-[#111]/10 rounded-full overflow-hidden mb-4">
                      <div
                        className="h-full bg-gradient-to-r from-[#F45B69] to-[#FF8A80] rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${progress}%` }}
                      ></div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-sm">
                        <div className="w-2 h-2 bg-[#F45B69] rounded-full animate-pulse"></div>
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-[#111]/80">
                          {processingMessage}
                        </p>
                        <p className="text-[10px] uppercase tracking-widest text-[#111]/40 mt-0.5">
                          {progress < 100
                            ? `${progress}% complete`
                            : "Finalizing..."}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {processedAudioUrl && !isProcessing && (
                <div className="mt-12 w-full max-w-md animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="flex justify-between items-center mb-3">
                    <label className="text-xs font-bold uppercase tracking-widest text-[#111]/40">
                      Processed Audio
                    </label>
                    <div className="h-px flex-1 bg-[#111]/10 ml-4"></div>
                  </div>
                  <div className="bg-[#EAE8E0] rounded-2xl p-4 flex items-center gap-4">
                    <button
                      onClick={togglePlayback}
                      className="w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-sm hover:shadow-md transition-all hover:scale-105 group"
                    >
                      {isPlaying ? (
                        <div className="flex gap-1">
                          <div className="w-1 h-5 bg-[#F45B69] rounded-full"></div>
                          <div className="w-1 h-5 bg-[#F45B69] rounded-full"></div>
                        </div>
                      ) : (
                        <div className="w-0 h-0 border-t-[8px] border-t-transparent border-l-[14px] border-l-[#F45B69] border-b-[8px] border-b-transparent ml-1"></div>
                      )}
                    </button>
                    <div className="flex-1">
                      <div className="flex items-center gap-1.5">
                        {[...Array(20)].map((_, i) => (
                          <div
                            key={i}
                            className={`w-1 rounded-full transition-all duration-150 ${
                              isPlaying ? "bg-[#F45B69]" : "bg-[#111]/20"
                            }`}
                            style={{
                              height: `${Math.random() * 20 + 8}px`,
                              animationDelay: `${i * 0.05}s`,
                            }}
                          ></div>
                        ))}
                      </div>
                      <p className="text-[10px] uppercase tracking-widest text-[#111]/40 mt-2 font-medium">
                        {isPlaying ? "Now Playing" : "Ready to Play"}
                      </p>
                    </div>
                  </div>
                  <audio
                    ref={audioRef}
                    src={processedAudioUrl}
                    onEnded={() => setIsPlaying(false)}
                    className="hidden"
                  />
                </div>
              )}

              {audioStats && !isProcessing && (
                <div className="mt-8 w-full max-w-md animate-in fade-in slide-in-from-bottom-6 duration-700 delay-100 pb-12">
                  <div className="flex justify-between items-center mb-4">
                    <label className="text-xs font-bold uppercase tracking-widest text-[#111]/40">
                      Performance Analysis
                    </label>
                    <div className="h-px flex-1 bg-[#111]/10 ml-4"></div>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <StatCard label="Estimated Key" value={audioStats.key} />
                    <StatCard
                      label="Tempo"
                      value={`${Math.round(audioStats.tempo)} BPM`}
                      sub={audioStats.tempo_stability}
                      highlight={audioStats.tempo_stability !== "Steady"}
                    />
                    <StatCard
                      label="Pitch Tendency"
                      value={audioStats.pitch_tendency}
                      sub={
                        audioStats.pitch_offset !== 0
                          ? `${audioStats.pitch_offset > 0 ? "+" : ""}${
                              audioStats.pitch_offset
                            } cents`
                          : "Perfect"
                      }
                      highlight={audioStats.pitch_tendency !== "In Tune"}
                    />
                    <StatCard
                      label="Style"
                      value={musicType.toUpperCase()}
                      sub="Auto-detected"
                    />
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <PastUploads onBack={() => setView("camera")} />
        )}
      </main>

      <style jsx global>{`
        @keyframes wave {
          0%,
          100% {
            height: 20%;
          }
          50% {
            height: 60%;
          }
        }
        .animate-wave {
          animation: wave 1s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
};

const MinimalSlider = ({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (val: number) => void;
}) => {
  return (
    <div className="flex flex-col items-center gap-4 group cursor-pointer">
      <div className="h-32 md:h-40 w-2 bg-[#EAE8E0] rounded-full relative">
        <div
          className="absolute bottom-0 left-0 right-0 bg-[#111] rounded-full transition-all duration-300 opacity-10 group-hover:opacity-20"
          style={{ height: `${(value / 5) * 100}%` }}
        ></div>

        <div
          className="absolute left-1/2 -translate-x-1/2 w-6 h-6 md:w-8 md:h-8 bg-white border-2 border-[#EAE8E0] rounded-full shadow-sm flex items-center justify-center transition-all duration-300 group-hover:scale-110 group-hover:border-[#F45B69]"
          style={{ bottom: `calc(${(value / 5) * 100}% - 16px)` }}
        >
          <div className="w-1.5 h-1.5 bg-[#F45B69] rounded-full"></div>
        </div>

        <div className="absolute inset-0 flex flex-col-reverse">
          {[1, 2, 3, 4, 5].map((step) => (
            <div
              key={step}
              onClick={() => onChange(step)}
              className="flex-1 w-8 -ml-3 z-10"
            />
          ))}
        </div>
      </div>
      <span className="text-[10px] font-bold uppercase tracking-widest text-[#111]/40 group-hover:text-[#111]/80 transition-colors">
        {label}
      </span>
    </div>
  );
};

const StatCard = ({
  label,
  value,
  sub,
  highlight = false,
}: {
  label: string;
  value: string | number;
  sub?: string;
  highlight?: boolean;
}) => (
  <div
    className={`p-4 rounded-xl border transition-all duration-300 ${
      highlight
        ? "bg-white border-[#F45B69]/20 shadow-sm"
        : "bg-[#EAE8E0] border-transparent"
    }`}
  >
    <p className="text-[10px] uppercase tracking-widest text-[#111]/40 font-bold mb-1">
      {label}
    </p>
    <p
      className={`text-lg font-bold leading-tight ${
        highlight ? "text-[#F45B69]" : "text-[#111]/80"
      }`}
    >
      {value}
    </p>
    {sub && (
      <p className="text-[10px] font-medium text-[#111]/40 mt-1">{sub}</p>
    )}
  </div>
);

export default Camera;

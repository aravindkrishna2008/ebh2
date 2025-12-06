"use client";
import { useRef, useEffect, useState } from "react";
import PastUploads from "./PastUploads";

const Camera = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [view, setView] = useState<'camera' | 'uploads'>('camera');
  const [isRecording, setIsRecording] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [voiceLevel, setVoiceLevel] = useState(3);
  const [beatsLevel, setBeatsLevel] = useState(3);
  const [noiseLevel, setNoiseLevel] = useState(3);
  const [musicType, setMusicType] = useState<"trap" | "pop" | "lofi">("lofi");

  // Refs to hold latest values for the async onstop callback
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
        const blob = new Blob(chunksRef.current, { type: "audio/mp3" });
        const formData = new FormData();
        formData.append("audio", blob, "recording.mp3");

        // Append current settings
        formData.append("musicType", musicTypeRef.current);
        formData.append("beatsLevel", beatsLevelRef.current.toString());
        formData.append("voiceLevel", voiceLevelRef.current.toString());
        formData.append("noiseLevel", noiseLevelRef.current.toString());

        try {
          await fetch("http://localhost:5000", {
            method: "POST",
            body: formData,
          });
          console.log("Audio and settings sent to server");
        } catch (error) {
          console.error("Error sending data:", error);
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

  return (
    <div className="min-h-screen w-full bg-[#F4F4F0] text-[#111] font-sans flex flex-col relative overflow-hidden selection:bg-[#F45B69] selection:text-white">
      {/* Header */}
      <header className="p-4 md:p-6 flex justify-between items-center border-b border-[#111]/10 bg-[#F4F4F0] z-50 relative">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-[#111] rounded-full flex items-center justify-center">
            <div className="w-3 h-3 bg-[#F45B69] rounded-full animate-pulse"></div>
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight lowercase leading-none">
              i<span className="text-[#F45B69]">o</span>
            </h1>
            <p className="text-[10px] uppercase tracking-widest opacity-50 font-medium">
              System 14
            </p>
          </div>
        </div>

        {/* Hamburger Button */}
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

        {/* Menu Overlay */}
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
                setView('uploads');
                setIsMenuOpen(false);
              }}
              className="text-3xl font-bold hover:text-[#F45B69] transition-colors"
            >
              Past Uploads
            </button>
            <button 
              onClick={() => {
                setView('camera');
                setIsMenuOpen(false);
              }}
              className="text-3xl font-bold hover:text-[#F45B69] transition-colors"
            >
              Log Out
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className="flex-1 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
        {view === 'camera' ? (
          <>
            {/* Left/Top: Camera Viewfinder */}
            <div className="flex-1 p-4 md:p-8 flex flex-col justify-center items-center relative bg-[#EAE8E0] lg:border-r border-[#111]/10 lg:min-h-0">
              <div className="relative w-full max-w-2xl aspect-[4/3] md:aspect-video bg-[#111] rounded-2xl overflow-hidden shadow-2xl ring-1 ring-black/5 group transition-all duration-500 hover:shadow-3xl">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover opacity-90 transition-opacity duration-500 group-hover:opacity-100"
                />

                {/* Audio Wave Animation Overlay */}
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

                {/* Minimalist Overlays */}
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

            {/* Right/Bottom: Controls */}
            <div className="flex-1 p-6 md:p-12 bg-[#F4F4F0] flex flex-col gap-8 md:gap-12 justify-center items-center relative">
              {/* Decorative Background Elements */}
              <div className="absolute top-10 right-10 w-20 h-20 border-t-2 border-r-2 border-[#111]/5 rounded-tr-3xl pointer-events-none"></div>
              <div className="absolute bottom-10 left-10 w-20 h-20 border-b-2 border-l-2 border-[#111]/5 rounded-bl-3xl pointer-events-none"></div>

              {/* Mode Selector */}
              <div className="w-full max-w-md space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-bold uppercase tracking-widest text-[#111]/40">
                    Tuning Mode
                  </label>
                  <div className="h-px flex-1 bg-[#111]/10 ml-4"></div>
                </div>
                <div className="flex gap-2 p-1 bg-[#EAE8E0] rounded-xl">
                  {["trap", "pop", "lofi"].map((type) => (
                    <button
                      key={type}
                      onClick={() => setMusicType(type as any)}
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

              {/* Sliders */}
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

              {/* Main Action */}
              <div className="mt-4 md:mt-8 relative group">
                <div
                  className={`absolute inset-0 bg-[#F45B69] rounded-full blur-xl opacity-20 transition-opacity duration-500 ${
                    isRecording ? "opacity-40 scale-150" : "group-hover:opacity-30"
                  }`}
                ></div>
                <button
                  onClick={toggleRecording}
                  className={`relative w-20 h-20 md:w-24 md:h-24 rounded-full border-[6px] transition-all duration-500 flex items-center justify-center z-10 ${
                    isRecording
                      ? "border-[#F45B69] bg-white rotate-180"
                      : "border-[#EAE8E0] bg-[#F45B69] hover:scale-105 hover:shadow-lg"
                  }`}
                >
                  <div
                    className={`transition-all duration-500 ${
                      isRecording
                        ? "w-8 h-8 bg-[#F45B69] rounded-sm"
                        : "w-3 h-3 bg-white rounded-full"
                    }`}
                  ></div>
                </button>
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-[10px] font-bold uppercase tracking-widest text-[#111]/30 whitespace-nowrap">
                  {isRecording ? "Tap to Stop" : "Tap to Record"}
                </div>
              </div>
            </div>
          </>
        ) : (
          <PastUploads onBack={() => setView('camera')} />
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
        {/* Track fill */}
        <div
          className="absolute bottom-0 left-0 right-0 bg-[#111] rounded-full transition-all duration-300 opacity-10 group-hover:opacity-20"
          style={{ height: `${(value / 5) * 100}%` }}
        ></div>

        {/* Handle */}
        <div
          className="absolute left-1/2 -translate-x-1/2 w-6 h-6 md:w-8 md:h-8 bg-white border-2 border-[#EAE8E0] rounded-full shadow-sm flex items-center justify-center transition-all duration-300 group-hover:scale-110 group-hover:border-[#F45B69]"
          style={{ bottom: `calc(${(value / 5) * 100}% - 16px)` }}
        >
          <div className="w-1.5 h-1.5 bg-[#F45B69] rounded-full"></div>
        </div>

        {/* Click areas */}
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

export default Camera;

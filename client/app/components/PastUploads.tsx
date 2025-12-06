import React from 'react';

const PastUploads = ({ onBack }: { onBack: () => void }) => {
  // Mock data
  const uploads = [
    { id: 1, date: '2024-12-06', time: '14:30', type: 'LOFI', duration: '0:45' },
    { id: 2, date: '2024-12-05', time: '09:15', type: 'TRAP', duration: '1:20' },
    { id: 3, date: '2024-12-04', time: '18:45', type: 'POP', duration: '0:30' },
    { id: 4, date: '2024-12-03', time: '11:20', type: 'LOFI', duration: '2:15' },
    { id: 5, date: '2024-12-01', time: '16:10', type: 'POP', duration: '1:05' },
  ];

  return (
    <div className="flex-1 w-full h-full p-6 md:p-12 overflow-y-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-center gap-4 mb-12">
            <button 
                onClick={onBack} 
                className="w-10 h-10 rounded-full border border-[#111]/10 flex items-center justify-center hover:bg-[#111] hover:text-white transition-all duration-300 group"
            >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="group-hover:-translate-x-0.5 transition-transform">
                    <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
            </button>
            <div>
                <h2 className="text-3xl font-bold tracking-tight leading-none">Library</h2>
                <p className="text-xs uppercase tracking-widest opacity-40 font-medium mt-1">Your Recordings</p>
            </div>
        </div>

        <div className="grid gap-4">
          {uploads.map((upload) => (
            <div key={upload.id} className="bg-[#EAE8E0] p-4 md:p-6 rounded-2xl flex items-center justify-between group hover:bg-white hover:shadow-xl hover:shadow-black/5 transition-all duration-300 cursor-pointer">
              <div className="flex items-center gap-4 md:gap-6">
                <div className="w-12 h-12 md:w-14 md:h-14 bg-[#F4F4F0] rounded-full flex items-center justify-center group-hover:bg-[#F45B69] transition-colors duration-300 shadow-sm">
                    <div className="w-0 h-0 border-t-[6px] border-t-transparent border-l-[10px] border-l-[#111] border-b-[6px] border-b-transparent ml-1 group-hover:border-l-white transition-colors"></div>
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full border ${
                        upload.type === 'LOFI' ? 'bg-blue-100 border-blue-200 text-blue-700' :
                        upload.type === 'TRAP' ? 'bg-purple-100 border-purple-200 text-purple-700' :
                        'bg-orange-100 border-orange-200 text-orange-700'
                    }`}>
                        {upload.type}
                    </span>
                    <span className="text-[10px] opacity-40 font-mono font-medium">{upload.duration}</span>
                  </div>
                  <div className="font-bold text-lg leading-none group-hover:text-[#F45B69] transition-colors">Recording {upload.id}</div>
                </div>
              </div>
              
              <div className="text-right hidden md:block">
                <div className="text-sm font-bold">{upload.date}</div>
                <div className="text-xs opacity-40 font-medium tracking-wider">{upload.time}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PastUploads;
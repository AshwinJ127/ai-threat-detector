import React, { useState, useEffect, useRef } from 'react';
import { Activity, Shield, Wifi, Terminal, Lock } from 'lucide-react';

const NetSentryDashboard = () => {
  // State for live data
  const [packets, setPackets] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [stats, setStats] = useState({ total: 0, malicious: 0, suspicious: 0, benign: 0 });
  
  // WebSocket Reference
  const ws = useRef(null);

  useEffect(() => {
    // --- CONNECT TO SERVER ---
    const SOCKET_URL = 'ws://localhost:8000/ws';
    ws.current = new WebSocket(SOCKET_URL);

    ws.current.onopen = () => {
      console.log('NetSentry Connection Established');
      setIsConnected(true);
    };

    ws.current.onclose = () => {
      console.log('NetSentry Connection Lost');
      setIsConnected(false);
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // Verify we have the metadata
      if (!data.packet_info) return;

      const newPacket = {
        id: Date.now(),
        // 1. Get the Details from the Simulator
        timestamp: data.packet_info.timestamp,
        src: data.packet_info.src_ip,
        dest: data.packet_info.dst_ip,
        protocol: data.packet_info.protocol,
        length: data.packet_info.length,
        
        // 2. Get the Prediction from the ML Model
        // (Assuming 1 = Malicious, 0 = Benign)
        severity: data.prediction === 1 ? 'high' : 'low',
        flag: data.prediction === 1 ? 'MALICIOUS' : 'ACK' 
      };

      // Update UI Lists
      setPackets((prev) => [newPacket, ...prev].slice(0, 50));
      
      setStats((prev) => ({
        total: prev.total + 1,
        malicious: data.prediction === 1 ? prev.malicious + 1 : prev.malicious,
        suspicious: prev.suspicious,
        benign: data.prediction === 0 ? prev.benign + 1 : prev.benign,
      }));
    };

    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-emerald-500 font-mono p-4 selection:bg-emerald-900 selection:text-white">
      
      {/* --- HEADER --- */}
      <header className="flex justify-between items-center border-b border-emerald-900/50 pb-4 mb-6">
        <div className="flex items-center gap-3">
          <Shield className="w-8 h-8 text-emerald-400" />
          <div>
            <h1 className="text-2xl font-bold tracking-wider text-white">NET_SENTRY</h1>
            <p className="text-xs text-emerald-600">INTRUSION DETECTION SYSTEM v1.0</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1 rounded text-xs font-bold border ${isConnected ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-400' : 'border-red-500/50 bg-red-500/10 text-red-400'}`}>
            <Wifi className="w-3 h-3" />
            {isConnected ? 'ONLINE' : 'OFFLINE'}
          </div>
        </div>
      </header>

      {/* --- MAIN GRID --- */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* LEFT COLUMN: STATS */}
        <aside className="lg:col-span-1 space-y-4">
          
          <div className="bg-slate-900/50 border border-emerald-900/50 p-4 rounded-sm">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-emerald-600" />
              <h3 className="text-sm font-bold text-slate-400">THREAT LEVEL</h3>
            </div>
            {/* Dynamic Threat Level */}
            <p className={`text-3xl font-bold ${stats.malicious > 0 ? 'text-red-500' : 'text-white'}`}>
              {stats.malicious > 0 ? 'CRITICAL' : 'NOMINAL'}
            </p>
          </div>

          <div className="bg-slate-900/50 border border-emerald-900/50 p-4 rounded-sm">
            <h3 className="text-xs text-slate-500 uppercase mb-4">Packet Analysis</h3>
            
            <div className="flex justify-between items-end mb-2">
              <span className="text-sm">Total Scanned</span>
              <span className="text-xl font-bold text-white">{stats.total}</span>
            </div>
            <div className="w-full bg-slate-800 h-1 mb-4">
              {/* Dynamic Progress Bar */}
              <div 
                className="bg-emerald-500 h-1 transition-all duration-300" 
                style={{ width: `${(stats.benign / (stats.total || 1)) * 100}%` }}
              ></div>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Malicious</span>
                <span className="text-red-400 font-bold">{stats.malicious}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Suspicious</span>
                <span className="text-yellow-400 font-bold">{stats.suspicious}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Benign</span>
                <span className="text-emerald-400 font-bold">{stats.benign}</span>
              </div>
            </div>
          </div>
        </aside>

        {/* RIGHT COLUMN: LOGS */}
        <main className="lg:col-span-3 bg-black border border-emerald-900/50 rounded-sm relative overflow-hidden flex flex-col h-[600px]">
          <div className="bg-emerald-900/20 px-4 py-2 border-b border-emerald-900/50 flex justify-between items-center">
            <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4" />
                <span className="text-sm font-bold">LIVE_TRAFFIC_STREAM</span>
            </div>
            {isConnected && (
              <div className="flex gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                  <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse delay-75"></span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse delay-150"></span>
              </div>
            )}
          </div>

          <div className="grid grid-cols-12 gap-2 px-4 py-2 text-xs text-slate-500 font-bold border-b border-emerald-900/30">
            <div className="col-span-1">TIME</div>
            <div className="col-span-3">SOURCE</div>
            <div className="col-span-3">DESTINATION</div>
            <div className="col-span-1">PROTO</div>
            <div className="col-span-1">LEN</div>
            <div className="col-span-2">FLAG</div>
            <div className="col-span-1 text-right">SEV</div>
          </div>

          <div className="flex-1 overflow-y-auto p-0 font-mono text-sm scrollbar-thin scrollbar-thumb-emerald-900 scrollbar-track-transparent">
            {packets.length === 0 ? (
               <div className="p-4 text-slate-600 italic">Waiting for traffic...</div>
            ) : (
                packets.map((packet, idx) => (
                <div key={`${packet.id}-${idx}`} className="grid grid-cols-12 gap-2 px-4 py-2 hover:bg-emerald-900/10 border-b border-emerald-900/10 transition-colors">
                    <div className="col-span-1 text-slate-400">{packet.timestamp}</div>
                    <div className="col-span-3 text-emerald-300">{packet.src}</div>
                    <div className="col-span-3 text-emerald-300">{packet.dest}</div>
                    <div className="col-span-1 text-slate-300">{packet.protocol}</div>
                    <div className="col-span-1 text-slate-500">{packet.length}</div>
                    <div className="col-span-2 text-slate-400">{packet.flag}</div>
                    <div className={`col-span-1 text-right font-bold ${
                    packet.severity === 'high' ? 'text-red-500' : 
                    packet.severity === 'medium' ? 'text-yellow-500' : 'text-slate-600'
                    }`}>
                    {packet.severity.toUpperCase()}
                    </div>
                </div>
                ))
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default NetSentryDashboard;
import { useState, useEffect } from 'react'
import { Brain, Zap, History, Play, ArrowRight } from 'lucide-react'
import ModelWatcher from './components/ModelWatcher'

const MODELS = [
  { id: 'dqn', name: 'DQN Baseline', desc: 'Deep Q-Network on 5x5 grid.', icon: Zap },
  { id: 'ppo', name: 'PPO Symbolic', desc: 'Proximal Policy Optimization with symbolic observations.', icon: Brain },
  { id: 'rppo_baseline', name: 'RPPO Baseline', desc: 'Recurrent PPO on 8x8 environments.', icon: History },
  { id: 'rppo_curriculum', name: 'RPPO Curriculum', desc: 'Staged training for complex navigation.', icon: Play },
]

const CustomCursor = () => {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const updatePosition = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    const updateHover = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      setIsHovering(
        target.tagName === 'BUTTON' || 
        target.tagName === 'A' || 
        target.closest('button') !== null ||
        target.closest('a') !== null
      );
    };

    window.addEventListener('mousemove', updatePosition);
    window.addEventListener('mouseover', updateHover);

    return () => {
      window.removeEventListener('mousemove', updatePosition);
      window.removeEventListener('mouseover', updateHover);
    };
  }, []);

  return (
    <div 
      className="custom-cursor-container" 
      style={{ transform: `translate3d(${position.x}px, ${position.y}px, 0)` }}
    >
      <div className={`custom-cursor-dot ${isHovering ? 'hovering' : ''}`} />
    </div>
  );
};

function App() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)

  return (
    <div className="min-h-screen bg-bg text-fg selection:bg-fg selection:text-bg">
      <CustomCursor />
      
      <div className="max-w-6xl mx-auto px-8 py-16">
        {/* Navigation / Header */}
        <nav className="flex justify-between items-baseline mb-24 border-b border-border pb-8">
          <div className="flex items-baseline gap-4">
            <h1 className="text-2xl m-0 leading-none">Agent Visualiser</h1>
            <span className="text-xs font-medium text-sub tracking-widest uppercase">0.1a</span>
          </div>
          <div className="flex gap-8 text-sm font-medium text-sub">
            <a href="https://github.com/nxck2005/rlp" className="hover:text-fg transition-colors">Source</a>
            <a href="https://github.com/nxck2005" className="hover:text-fg transition-colors">GitHub</a>
          </div>
        </nav>

        {/* Hero Section */}
        {!selectedModel && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-1000">
            <h2 className="text-6xl mb-12 max-w-2xl leading-[1.1]">
              Visualise agent walkthroughs for RL agents;
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-24">
              {MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className="group relative text-left p-8 border border-border hover:border-fg transition-all duration-500 bg-bg"
                >
                  <div className="flex justify-between items-start mb-12">
                    <model.icon size={24} strokeWidth={1.5} className="text-sub group-hover:text-fg transition-colors" />
                    <ArrowRight size={20} className="text-border group-hover:text-fg transform -rotate-45 group-hover:rotate-0 transition-all" />
                  </div>
                  <h3 className="text-xl mb-2 m-0">{model.name}</h3>
                  <p className="text-sm text-sub m-0 max-w-[240px] leading-relaxed">
                    {model.desc}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Lab Workspace */}
        {selectedModel && (
          <div className="animate-in fade-in duration-700">
            <div className="flex justify-between items-center mb-8">
              <button 
                onClick={() => setSelectedModel(null)}
                className="text-sm text-sub hover:text-fg flex items-center gap-2 group transition-colors"
              >
                <ArrowRight size={16} className="rotate-180 group-hover:-translate-x-1 transition-transform" />
                Return to Index
              </button>
              <div className="flex gap-4">
                {MODELS.map(m => (
                  <button 
                    key={m.id}
                    onClick={() => setSelectedModel(m.id)}
                    className={`w-2 h-2 rounded-full border border-fg transition-all ${selectedModel === m.id ? 'bg-fg scale-125' : 'bg-transparent opacity-30 hover:opacity-100'}`}
                    title={m.name}
                  />
                ))}
              </div>
            </div>
            
            <div className="w-full min-h-[700px]">
              <ModelWatcher 
                key={selectedModel}
                modelId={selectedModel} 
                name={MODELS.find(m => m.id === selectedModel)?.name || ''} 
              />
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-32 pt-12 border-t border-border flex justify-between items-center text-[10px] uppercase tracking-[0.2em] text-sub">
          <span>Ran on React 19 and Vite</span>
          <div className="flex gap-8">
            <span>Made for CSE4037</span>
            <span>J Component</span>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App

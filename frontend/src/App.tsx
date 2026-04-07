import { useState, useEffect } from 'react'
import { Brain, Zap, History, Play, ArrowRight, Layers, Target, Cpu } from 'lucide-react'
import ModelWatcher from './components/ModelWatcher'

interface ModelOption {
  id: string;
  name: string;
  desc: string;
}

interface Approach {
  id: string;
  title: string;
  subtitle: string;
  icon: any;
  variants: ModelOption[];
}

const APPROACHES: Approach[] = [
  {
    id: 'dqn',
    title: 'DQN ARCHITECTURES',
    subtitle: 'Deep Q-Networks / Pixel-Based',
    icon: Zap,
    variants: [
      { id: 'dqn_baseline', name: 'Standard Baseline', desc: 'Raw DQN on 5x5 grid without temporal context.' },
      { id: 'dqn_framestack', name: 'Frame Stacking', desc: 'Adds 4-frame buffer to provide short-term memory.' },
      { id: 'dqn_custom_cnn', name: 'Custom Minigrid CNN', desc: 'Optimised kernel sizes for 56x56 sparse observations.' },
      { id: 'dqn_curriculum', name: 'Staged DQN', desc: 'Transfer learning across progressively harder maps.' },
    ]
  },
  {
    id: 'ppo_pixel',
    title: 'PPO VISUAL',
    subtitle: 'Pixel-Based Proximal Policy Optimization',
    icon: Target,
    variants: [
      { id: 'ppo_baseline', name: 'Visual Baseline', desc: 'On-policy stability for raw pixel observations.' },
      { id: 'ppo_curriculum', name: 'Visual Curriculum', desc: 'Staged navigation training for vision-based agents.' },
    ]
  },
  {
    id: 'ppo_symbolic',
    title: 'PPO SYMBOLIC',
    subtitle: 'Fast Convergence / Logical Extraction',
    icon: Brain,
    variants: [
      { id: 'ppo_flat', name: 'Flat Observed', desc: 'Direct state extraction (Object IDs) for rapid learning.' },
      { id: 'ppo_flat_cur', name: 'Flat Curriculum', desc: 'Logical transfer across symbolic environment stages.' },
    ]
  },
  {
    id: 'rppo',
    title: 'RECURRENT PPO',
    subtitle: 'LSTM-Driven / POMDP Solutions',
    icon: History,
    variants: [
      { id: 'rppo_baseline', name: 'Recurrent 8x8', desc: 'LSTM memory for 8x8 maps where goals are hidden.' },
      { id: 'rppo_curriculum', name: 'Recurrent Curriculum', desc: '3-stage transfer learning for complex spatial logic.' },
    ]
  },
  {
    id: 'repp2',
    title: 'REPP2 ADVANCED',
    subtitle: 'Parallelized / Multi-Stage Sequence',
    icon: Cpu,
    variants: [
      { id: 'repp2_4stage', name: '4-Stage Parallel', desc: 'SubprocVecEnv accelerated training across 4 CPU cores.' },
    ]
  }
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
  const [expandedApproach, setExpandedApproach] = useState<string | null>(null)

  const currentModelData = APPROACHES.flatMap(a => a.variants).find(v => v.id === selectedModel);

  return (
    <div className="min-h-screen bg-bg text-fg selection:bg-fg selection:text-bg">
      <CustomCursor />
      
      <div className="max-w-6xl mx-auto px-8 py-16">
        {/* Navigation / Header */}
        <nav className="flex justify-between items-baseline mb-24 border-b border-border pb-8">
          <div className="flex items-baseline gap-4">
            <h1 className="text-2xl m-0 font-heading">Agent Visualiser</h1>
            <span className="text-[10px] font-bold text-sub tracking-[0.3em] uppercase">Phase_03</span>
          </div>
          <div className="flex gap-8 text-[10px] font-bold uppercase tracking-widest text-sub">
            <a href="https://github.com/nxck2005/rlp" className="hover:text-fg transition-colors">Source_Code</a>
            <a href="https://github.com/nxck2005" className="hover:text-fg transition-colors">Archive</a>
          </div>
        </nav>

        {/* Hero Section / Approach Selection */}
        {!selectedModel && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-1000">
            <h2 className="text-5xl mb-16 max-w-2xl leading-[1.1] font-heading lowercase tracking-tighter">
              A study of <span className="text-sub">Curriculum Learning</span> versus <span className="text-sub">Baseline Training</span>;
            </h2>
            
            <div className="space-y-4 mt-24">
              {APPROACHES.map((approach) => (
                <div key={approach.id} className="border border-border bg-bg overflow-hidden transition-all duration-500">
                  <button
                    onClick={() => setExpandedApproach(expandedApproach === approach.id ? null : approach.id)}
                    className={`w-full flex items-center justify-between p-8 hover:bg-accent-bg transition-colors ${expandedApproach === approach.id ? 'bg-accent-bg' : ''}`}
                  >
                    <div className="flex items-center gap-8">
                      <approach.icon size={20} strokeWidth={1.5} className="text-sub" />
                      <div className="text-left">
                        <h3 className="text-xs font-bold tracking-[0.2em] mb-1">{approach.title}</h3>
                        <p className="text-[10px] text-sub uppercase tracking-widest">{approach.subtitle}</p>
                      </div>
                    </div>
                    <ArrowRight size={20} className={`text-sub transition-transform duration-500 ${expandedApproach === approach.id ? 'rotate-90' : '-rotate-45'}`} />
                  </button>

                  <div 
                    className={`grid transition-all duration-500 ease-in-out ${expandedApproach === approach.id ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}
                  >
                    <div className="overflow-hidden">
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-px bg-border border-t border-border">
                        {approach.variants.map((variant) => (
                          <button
                            key={variant.id}
                            onClick={() => setSelectedModel(variant.id)}
                            className="p-6 bg-bg hover:bg-accent-bg text-left transition-colors group h-full flex flex-col justify-between"
                          >
                            <div>
                              <h4 className="text-[11px] font-bold mb-3 uppercase tracking-wider">{variant.name}</h4>
                              <p className="text-[10px] text-sub leading-relaxed lowercase mb-6">{variant.desc}</p>
                            </div>
                            <span className="text-[10px] font-bold tracking-tighter group-hover:translate-x-1 transition-transform inline-flex items-center gap-2">
                              Launch Session <ArrowRight size={10} />
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Lab Workspace */}
        {selectedModel && (
          <div className="animate-in fade-in duration-700">
            <div className="flex justify-between items-center mb-12 border-b border-border pb-8">
              <button 
                onClick={() => setSelectedModel(null)}
                className="text-[10px] font-bold tracking-widest text-sub hover:text-fg flex items-center gap-2 group transition-colors uppercase"
              >
                <ArrowRight size={14} className="rotate-180 group-hover:-translate-x-1 transition-transform" />
                Index_Navigation
              </button>
              <div className="text-right">
                <span className="text-[10px] text-sub uppercase tracking-[0.2em] block mb-2">Active_Model</span>
                <span className="text-sm font-bold uppercase tracking-widest">{currentModelData?.name}</span>
              </div>
            </div>
            
            <div className="w-full min-h-[700px]">
              <ModelWatcher 
                key={selectedModel}
                modelId={selectedModel} 
                name={currentModelData?.name || ''} 
              />
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-32 pt-12 border-t border-border flex justify-between items-center text-[10px] uppercase tracking-[0.2em] text-sub">
          <span>React 19 x SB3 Contrib</span>
          <div className="flex gap-8">
            <span>Laboratory Environment</span>
            <span>2026_Edition</span>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App

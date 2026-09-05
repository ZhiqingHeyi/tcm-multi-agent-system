import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ArrowRight,
  ArrowLeft,
  RefreshCw,
  Sparkles,
  Settings,
  BrainCircuit,
  MessageSquarePlus,
  Stethoscope,
  ChevronRight,
  ChevronLeft,
  Feather,
  CheckCircle2
} from 'lucide-react'
import {
  api,
  type School,
  type QuestionModule,
  type Question,
  type FinalReport,
  type AgentOpinion,
  type LLMStatus,
  streamReport
} from './api'

type View = 'home' | 'selection' | 'questionnaire' | 'followup' | 'analyzing' | 'report' | 'admin'

export function App() {
  const [view, setView] = useState<View>('home')
  const [schools, setSchools] = useState<School[]>([])
  const [selectedSchool, setSelectedSchool] = useState<School | null>(null)
  const [sessionId, setSessionId] = useState('')

  // 题库与问诊
  const [modules, setModules] = useState<QuestionModule[]>([])
  const [currentModIdx, setCurrentModIdx] = useState(0)
  const [currentQIdx, setCurrentQIdx] = useState(0)
  const [facts, setFacts] = useState<Record<string, string>>({})

  // AI 动态追问
  const [followupQuestions, setFollowupQuestions] = useState<{ title: string; key: string; options: string[] }[]>([])
  const [followupIdx, setFollowupIdx] = useState(0)

  // 推演分析过程 (SSE)
  const [stageMsg, setStageMsg] = useState('')
  const [panelOpinions, setPanelOpinions] = useState<AgentOpinion[]>([])
  const [finalReport, setFinalReport] = useState<FinalReport | null>(null)
  const [activeTab, setActiveTab] = useState<'final' | 'panel'>('final')

  // 管理员配置
  const [llmStatus, setLlmStatus] = useState<LLMStatus | null>(null)
  const [adminToken, setAdminToken] = useState('tcm-admin')
  const [adminBaseUrl, setAdminBaseUrl] = useState('')
  const [adminApiKey, setAdminApiKey] = useState('')
  const [adminModelFast, setAdminModelFast] = useState('')
  const [adminModelPro, setAdminModelPro] = useState('')
  const [adminMsg, setAdminMsg] = useState('')
  const [adminLoading, setAdminLoading] = useState(false)

  useEffect(() => {
    api.agents().then(setSchools).catch(() => undefined)
    api.questionnaire().then((res) => setModules(res.modules)).catch(() => undefined)
    api.llmStatus().then((res) => {
      setLlmStatus(res)
      setAdminBaseUrl(res.llm_base_url)
      setAdminModelFast(res.llm_model_fast)
      setAdminModelPro(res.llm_model_pro)
    }).catch(() => undefined)
  }, [])

  const startConsultation = async (school: School) => {
    setSelectedSchool(school)
    setFacts({})
    setCurrentModIdx(0)
    setCurrentQIdx(0)
    try {
      const res = await api.createConsultation(school.id)
      setSessionId(res.id)
      setView('questionnaire')
    } catch {
      alert('创建问诊会话失败')
    }
  }

  const currentModule = modules[currentModIdx]
  const currentQuestion: Question | undefined = currentModule?.questions[currentQIdx]

  const handleSelectOption = (opt: string) => {
    if (!currentQuestion) return
    const nextFacts = { ...facts, [currentQuestion.title]: opt }
    setFacts(nextFacts)

    if (currentQIdx + 1 < currentModule.questions.length) {
      setCurrentQIdx(currentQIdx + 1)
    } else if (currentModIdx + 1 < modules.length) {
      setCurrentModIdx(currentModIdx + 1)
      setCurrentQIdx(0)
    } else {
      finishQuestionnaire(nextFacts)
    }
  }

  const handlePrevQuestion = () => {
    if (currentQIdx > 0) {
      setCurrentQIdx(currentQIdx - 1)
    } else if (currentModIdx > 0) {
      const prevMod = modules[currentModIdx - 1]
      setCurrentModIdx(currentModIdx - 1)
      setCurrentQIdx(prevMod.questions.length - 1)
    }
  }

  const finishQuestionnaire = async (finalFacts: Record<string, string>) => {
    setStageMsg('正在呈递主控脑枢，研判关键脉络…')
    setView('analyzing')
    try {
      await api.saveAnswers(sessionId, finalFacts)
      const fu = await api.followup(sessionId)
      if (fu.questions && fu.questions.length > 0) {
        setFollowupQuestions(fu.questions)
        setFollowupIdx(0)
        setView('followup')
      } else {
        startPanelAnalysis(finalFacts)
      }
    } catch {
      startPanelAnalysis(finalFacts)
    }
  }

  const handleFollowupOption = (opt: string) => {
    const q = followupQuestions[followupIdx]
    const nextFacts = { ...facts, [q.title]: opt }
    setFacts(nextFacts)

    if (followupIdx + 1 < followupQuestions.length) {
      setFollowupIdx(followupIdx + 1)
    } else {
      startPanelAnalysis(nextFacts)
    }
  }

  const startPanelAnalysis = (allFacts: Record<string, string>) => {
    setView('analyzing')
    setPanelOpinions([])
    setFinalReport(null)

    streamReport(sessionId, allFacts, {
      onStage: (msg) => setStageMsg(msg),
      onAgentResult: (opinion) => setPanelOpinions((prev) => [...prev, opinion]),
      onReport: (report) => {
        setFinalReport(report)
        setView('report')
      },
      onError: (err) => {
        alert(`推演受阻: ${err.message}`)
        setView('questionnaire')
      }
    })
  }

  const saveAdminSettings = async () => {
    setAdminLoading(true)
    setAdminMsg('')
    try {
      const res = await api.updateLLM(
        {
          base_url: adminBaseUrl,
          api_key: adminApiKey,
          model_fast: adminModelFast,
          model_pro: adminModelPro
        },
        adminToken
      )
      setLlmStatus(res)
      setAdminMsg('配置保存成功！')
    } catch (e: any) {
      setAdminMsg(`保存失败: ${e.message}`)
    } finally {
      setAdminLoading(false)
    }
  }

  const testAdminConnection = async () => {
    setAdminLoading(true)
    setAdminMsg('正在测试模型连通性…')
    try {
      const res = await api.testLLM(adminToken, 'both')
      if (res.ok) {
        setAdminMsg(`连通成功！Flash敏捷核: ${res.replies.fast} | Pro辨证核: ${res.replies.pro}`)
      }
    } catch (e: any) {
      setAdminMsg(`测试报错: ${e.message}`)
    } finally {
      setAdminLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#EADBB8] bg-paper text-[#27251F] selection:bg-[#9B4B3E]/20 flex flex-col justify-between font-serif relative">
      <div className="paper-grain" />

      {/* 顶部古雅线装书卷标头 */}
      <header className="border-b border-[#D8C7A0] px-8 py-5 flex items-center justify-between bg-[#EFE4C8]/70 backdrop-blur-sm sticky top-0 z-40">
        <div className="flex items-center gap-4 cursor-pointer" onClick={() => setView('home')}>
          <div className="seal text-lg font-bold border-2">岐</div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-xl tracking-widest text-[#27251F]">本草问心</h1>
              <span className="text-[11px] px-2 py-0.5 border border-[#385B62]/40 text-[#385B62] rounded-sm">
                古籍宣纸卷
              </span>
            </div>
            <p className="text-[11px] text-[#786650] tracking-[0.2em] font-sans">
              多AGENT名医学派 · 辩证论治案
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center gap-2 text-xs border border-[#385B62]/30 px-3.5 py-1 bg-[#F5EED9] text-[#385B62]">
            <span className={`w-2 h-2 rounded-full ${llmStatus?.configured ? 'bg-[#385B62]' : 'bg-[#AA7B43]'}`} />
            <span>{llmStatus?.configured ? '名医AI双核待命' : '规则底座兜底'}</span>
          </div>
          <button
            onClick={() => setView('admin')}
            className="seal !w-8 !h-8 hover:bg-[#9B4B3E] hover:text-white transition-colors"
            title="管治枢纽"
          >
            枢
          </button>
        </div>
      </header>

      {/* 正文中枢 */}
      <main className="flex-1 max-w-5xl w-full mx-auto p-6 md:p-10 flex flex-col justify-center relative z-10">
        <AnimatePresence mode="wait">
          {/* 首页：竖排书卷 + 苍劲东方墨韵 */}
          {view === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="py-12 flex flex-col items-center justify-center text-center space-y-10"
            >
              {/* 竖排对联装饰与主题 */}
              <div className="flex items-center justify-center gap-10 md:gap-14 my-4">
                <div className="writing-vertical text-xs tracking-[0.4em] text-[#7D6B55] border-r border-[#D2BF95] pr-4 select-none">
                  上工治未病 · 察毫芒于未萌
                </div>

                <div className="space-y-6 max-w-xl">
                  <div className="inline-block seal !w-auto !h-auto px-4 py-1.5 !rotate-0 text-sm tracking-widest font-semibold">
                    御览四诊 · 辨证推求
                  </div>
                  <h2 className="text-4xl md:text-6xl font-extrabold tracking-[0.18em] text-[#27251F] leading-tight font-serif">
                    古法十问<br />
                    <span className="text-[#9B4B3E]">学派合参</span>
                  </h2>
                  <p className="text-sm md:text-base text-[#5E4F3E] leading-relaxed tracking-wider">
                    融汇伤寒六经、温病卫气营血、东垣脾胃升降、火神重阳扶阳、汇通中西互证。
                    系统十问，智能随问，五大师承Agent同台辩难，共拟方药治则。
                  </p>
                </div>

                <div className="writing-vertical text-xs tracking-[0.4em] text-[#7D6B55] border-l border-[#D2BF95] pl-4 select-none">
                  见微以知著 · 审病机而立意
                </div>
              </div>

              <div className="pt-4">
                <button
                  onClick={() => setView('selection')}
                  className="ink-button text-base tracking-widest font-medium shadow-paper"
                >
                  <span>启卷 · 选择引诊先生</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </motion.div>
          )}

          {/* 学派选择：古籍名医册页 */}
          {view === 'selection' && (
            <motion.div
              key="selection"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-8"
            >
              <div className="text-center space-y-2 border-b border-[#D8C7A0] pb-6">
                <div className="seal !w-auto !h-auto px-3 py-1 text-xs mx-auto mb-2">五脉流芳</div>
                <h3 className="text-3xl font-bold tracking-widest text-[#27251F]">择定引诊先生</h3>
                <p className="text-xs text-[#786650] tracking-wider">
                  先生将引领四诊叩问。至终局推演，五大学派皆会一同登堂会诊、各陈方策。
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {schools.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => startConsultation(s)}
                    className="group border border-[#D8C7A0] bg-[#F7EED9]/60 hover:bg-[#FFFBF0] p-6 cursor-pointer shadow-paper transition-all relative overflow-hidden flex flex-col justify-between"
                  >
                    <div className="border-b border-[#D8C7A0]/70 pb-3 mb-4 flex items-center justify-between">
                      <span className="text-xs font-semibold px-2.5 py-0.5 border border-[#9B4B3E] text-[#9B4B3E]">
                        {s.title}
                      </span>
                      <span className="text-[11px] text-[#8C7A65] tracking-widest font-sans">学派正宗</span>
                    </div>

                    <div className="space-y-2 mb-6">
                      <h4 className="text-2xl font-bold text-[#27251F] group-hover:text-[#9B4B3E] transition-colors">
                        {s.name}
                      </h4>
                      <p className="text-xs font-medium text-[#385B62] tracking-wider">{s.specialty}</p>
                      <p className="text-xs text-[#6A5A48] leading-relaxed pt-1 font-serif">{s.style}</p>
                    </div>

                    <div className="flex items-center justify-between pt-3 border-t border-[#D8C7A0]/70 text-xs text-[#786650]">
                      <span>入室问诊</span>
                      <span className="group-hover:translate-x-1 transition-transform">→</span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* 十问问诊页：仿古籍折页 (左侧答题札记 + 右侧立式医案簿) */}
          {view === 'questionnaire' && currentQuestion && (
            <motion.div
              key="questionnaire"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-8"
            >
              {/* 左侧：问答答题折页 */}
              <div className="lg:col-span-2 border-2 border-[#D8C7A0] bg-[#F8F1DE]/90 p-8 shadow-paper flex flex-col justify-between relative">
                {/* 古籍四角装饰 */}
                <div className="absolute top-2 left-2 text-[#C0AE88] text-[10px]">「</div>
                <div className="absolute top-2 right-2 text-[#C0AE88] text-[10px]">」</div>
                <div className="absolute bottom-2 left-2 text-[#C0AE88] text-[10px]">『</div>
                <div className="absolute bottom-2 right-2 text-[#C0AE88] text-[10px]">』</div>

                <div>
                  {/* 分卷标识 */}
                  <div className="flex items-center justify-between border-b border-[#D8C7A0] pb-3 mb-6 text-xs text-[#7B6852]">
                    <div className="flex items-center gap-2">
                      <span className="seal !w-6 !h-6 !text-[10px]">卷</span>
                      <span className="font-bold text-sm tracking-widest text-[#27251F]">
                        第 {currentModIdx + 1} 卷 · {currentModule.module}
                      </span>
                    </div>
                    <span className="font-sans text-[11px] text-[#8C7A65]">
                      第 {currentQIdx + 1} 条 / 共 {currentModule.questions.length} 条
                    </span>
                  </div>

                  {/* 问诊条目 */}
                  <div className="mb-6">
                    <span className="text-xs text-[#9B4B3E] font-semibold tracking-widest block mb-1">【审察四诊】</span>
                    <h3 className="text-2xl md:text-3xl font-extrabold text-[#27251F] leading-snug tracking-wide">
                      {currentQuestion.title}
                    </h3>
                  </div>

                  {/* 选项矩阵：仿名牌签条 */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 mb-8">
                    {currentQuestion.options.map((opt) => {
                      const isSelected = facts[currentQuestion.title] === opt
                      return (
                        <button
                          key={opt}
                          onClick={() => handleSelectOption(opt)}
                          className={`option-card rounded-none border ${
                            isSelected
                              ? '!bg-[#385B62] !text-[#FDF9EE] !border-[#385B62] shadow-sm font-semibold'
                              : ''
                          }`}
                        >
                          <span className="text-sm md:text-base tracking-wider">{opt}</span>
                          <span className={`text-xs ${isSelected ? 'text-[#FDF9EE]' : 'text-[#8C7A65]'}`}>
                            {isSelected ? '已录' : '择定'}
                          </span>
                        </button>
                      )
                    })}
                  </div>
                </div>

                {/* 底部卷轴控制 */}
                <div className="flex items-center justify-between pt-5 border-t border-[#D8C7A0] text-xs">
                  <button
                    onClick={handlePrevQuestion}
                    disabled={currentModIdx === 0 && currentQIdx === 0}
                    className="text-[#685642] hover:text-[#27251F] disabled:opacity-30 inline-flex items-center gap-1.5"
                  >
                    <ChevronLeft className="w-4 h-4" /> 翻回上一页
                  </button>

                  <button
                    onClick={() => handleSelectOption('不确定')}
                    className="text-[#8C7A65] hover:text-[#9B4B3E] underline decoration-dotted"
                  >
                    暂不确定，留白备考
                  </button>

                  <button
                    onClick={() => finishQuestionnaire(facts)}
                    className="px-4 py-2 border border-[#9B4B3E] text-[#9B4B3E] hover:bg-[#9B4B3E] hover:text-white transition-all font-semibold tracking-wider"
                  >
                    定策 · 提前呈递推演
                  </button>
                </div>
              </div>

              {/* 右侧：宣纸竖线医案札记簿 */}
              <div className="border border-[#D8C7A0] bg-[#F7EED9]/80 p-6 shadow-paper flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b-2 border-[#27251F] pb-3 mb-4">
                    <div className="flex items-center gap-2">
                      <div className="seal !w-6 !h-6 !text-xs">案</div>
                      <h4 className="font-bold text-base tracking-widest text-[#27251F]">四诊随笔簿</h4>
                    </div>
                    <span className="text-[10px] text-[#8C7A65] font-sans">
                      已纳 {Object.keys(facts).length} 症
                    </span>
                  </div>

                  <div className="overflow-y-auto max-h-[420px] space-y-2.5 pr-1 text-xs">
                    {Object.keys(facts).length === 0 ? (
                      <p className="text-[#998770] italic text-center py-16 tracking-widest">
                        虚室以待 · 待书四诊
                      </p>
                    ) : (
                      Object.entries(facts).map(([k, v]) => (
                        <div
                          key={k}
                          className="bg-[#FFFBF2] p-3 border-l-2 border-[#385B62] border-t border-r border-b border-[#E6D7B5] flex flex-col gap-1 shadow-sm"
                        >
                          <span className="text-[11px] text-[#7A6854]">{k}</span>
                          <span className="font-bold text-[#27251F] text-sm tracking-wide">{v}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                <div className="pt-4 border-t border-[#D8C7A0] text-[11px] text-[#7A6854] leading-relaxed">
                  ※ 随问随录，条分缕析；问卷完成后将由五大师派共同登堂辨证。
                </div>
              </div>
            </motion.div>
          )}

          {/* AI 动态追问 (主控先生精准探求) */}
          {view === 'followup' && followupQuestions[followupIdx] && (
            <motion.div
              key="followup"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="max-w-2xl mx-auto w-full border-2 border-[#9B4B3E] bg-[#F8F1DE] p-8 md:p-10 shadow-paper"
            >
              <div className="flex items-center justify-between border-b border-[#D8C7A0] pb-3 mb-6">
                <div className="flex items-center gap-2">
                  <div className="seal !w-7 !h-7 !text-xs !bg-[#9B4B3E] !text-white !border-[#9B4B3E]">探</div>
                  <span className="font-bold text-sm tracking-widest text-[#9B4B3E]">
                    主控大医 · 精准追询 ({followupIdx + 1} / {followupQuestions.length})
                  </span>
                </div>
                <span className="text-xs text-[#7B6852]">补足关键病机证据</span>
              </div>

              <h3 className="text-2xl md:text-3xl font-bold text-[#27251F] mb-6 leading-relaxed">
                {followupQuestions[followupIdx].title}
              </h3>

              <div className="space-y-3.5 mb-8">
                {followupQuestions[followupIdx].options.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => handleFollowupOption(opt)}
                    className="option-card w-full rounded-none !bg-white/70 hover:!bg-[#F5E7CA] text-base"
                  >
                    <span className="font-medium tracking-wide">{opt}</span>
                    <span className="text-xs text-[#8C7A65]">确认录入 →</span>
                  </button>
                ))}
              </div>

              <div className="flex justify-end pt-2 border-t border-[#D8C7A0]">
                <button
                  onClick={() => startPanelAnalysis(facts)}
                  className="text-xs text-[#7B6852] hover:text-[#9B4B3E] underline decoration-dotted"
                >
                  无须深究，直接登堂会诊 →
                </button>
              </div>
            </motion.div>
          )}

          {/* 实时推演动画 (卷轴收紧，众医各抒己见) */}
          {view === 'analyzing' && (
            <motion.div
              key="analyzing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-20 space-y-8 max-w-lg mx-auto"
            >
              <div className="seal !w-16 !h-16 !text-2xl mx-auto animate-pulse border-2">
                参
              </div>
              <div className="space-y-3">
                <h3 className="text-2xl font-extrabold tracking-widest text-[#27251F]">
                  诸家名医 · 登堂推演
                </h3>
                <p className="text-xs text-[#6B5A46] tracking-widest">
                  {stageMsg || '伤寒、温病、脾胃、火神、汇通五派深研病机中…'}
                </p>
              </div>

              {panelOpinions.length > 0 && (
                <div className="text-left border border-[#D8C7A0] bg-[#FFFDF7] p-5 shadow-paper space-y-2.5 text-xs">
                  <div className="font-bold text-[#27251F] border-b border-[#E6D7B5] pb-2 flex justify-between">
                    <span>已奉札记</span>
                    <span className="text-[#385B62]">五派辨证</span>
                  </div>
                  {panelOpinions.map((o) => (
                    <div key={o.school} className="flex items-center justify-between text-[#5C4C3B] pt-1">
                      <span className="font-medium">{o.title} · {o.name}</span>
                      <span className="text-[#9B4B3E] font-bold tracking-wider">{o.diagnosis}</span>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {/* 辨证终局古籍案札 (双栏排版 + 印章) */}
          {view === 'report' && finalReport && (
            <motion.div
              key="report"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-8"
            >
              {/* 案札顶栏与切换标签 */}
              <div className="flex flex-col md:flex-row md:items-end justify-between border-b-2 border-[#27251F] pb-4 gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <div className="seal !w-8 !h-8 !text-sm !border-[#9B4B3E] !text-[#9B4B3E]">定</div>
                    <h3 className="text-3xl font-extrabold tracking-[0.2em] text-[#27251F]">中医辨证论治札</h3>
                  </div>
                  <p className="text-xs text-[#7A6854] tracking-widest">
                    五派各陈方略 · 主控综核审定 · 仅供医师复核
                  </p>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={() => setActiveTab('final')}
                    className={`px-5 py-2.5 text-xs font-bold tracking-widest border transition-all ${
                      activeTab === 'final'
                        ? 'bg-[#385B62] text-[#FDF9EE] border-[#385B62] shadow-sm'
                        : 'bg-[#F7EED9] text-[#4A3D2E] border-[#D8C7A0]'
                    }`}
                  >
                    合参统宗案
                  </button>
                  <button
                    onClick={() => setActiveTab('panel')}
                    className={`px-5 py-2.5 text-xs font-bold tracking-widest border transition-all ${
                      activeTab === 'panel'
                        ? 'bg-[#385B62] text-[#FDF9EE] border-[#385B62] shadow-sm'
                        : 'bg-[#F7EED9] text-[#4A3D2E] border-[#D8C7A0]'
                    }`}
                  >
                    五派学术辨 ({finalReport.panel?.length || 0})
                  </button>
                </div>
              </div>

              {/* 标签页内容 */}
              {activeTab === 'final' ? (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  {/* 左2列：宗案核心断决 */}
                  <div className="lg:col-span-2 space-y-6">
                    {/* 核心主证 */}
                    <div className="border-2 border-[#9B4B3E] bg-[#FAF3E3] p-8 shadow-paper relative">
                      <div className="absolute top-4 right-4 seal !w-12 !h-12 !text-xs !rotate-12 !border-[#9B4B3E] !text-[#9B4B3E]">
                        确诊印
                      </div>

                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-bold tracking-widest text-[#9B4B3E]">【统参审定证型】</span>
                        <span className="text-xs text-[#786650] font-sans">
                          置信度 {(finalReport.confidence * 100).toFixed(0)}%
                        </span>
                      </div>

                      <h2 className="text-4xl font-extrabold tracking-wider text-[#9B4B3E] mb-6 font-serif">
                        {finalReport.diagnosis}
                      </h2>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-[#E0D0B0] pt-5">
                        <div>
                          <p className="text-xs text-[#7D6B55] font-semibold mb-1">【治则治法】</p>
                          <p className="text-base font-bold text-[#27251F]">{finalReport.treatment}</p>
                        </div>
                        <div>
                          <p className="text-xs text-[#7D6B55] font-semibold mb-1">【方药参酌（须医师复核）】</p>
                          <p className="text-base font-bold text-[#9B4B3E]">{finalReport.formula || '不宜独断，宜面诊合参'}</p>
                        </div>
                      </div>

                      {finalReport.modifications && (
                        <div className="mt-4 pt-4 border-t border-[#E0D0B0]">
                          <p className="text-xs text-[#7D6B55] font-semibold mb-1">【随症加减意向】</p>
                          <p className="text-xs text-[#524434] leading-relaxed">{finalReport.modifications}</p>
                        </div>
                      )}
                    </div>

                    {/* 病机阐微 */}
                    <div className="border border-[#D8C7A0] bg-[#FFFBF2] p-7 shadow-paper space-y-2">
                      <h4 className="font-bold text-sm tracking-widest text-[#27251F] border-b border-[#E6D7B5] pb-2">
                        【病机阐微】
                      </h4>
                      <p className="text-sm text-[#4F4132] leading-relaxed tracking-wide pt-1">
                        {finalReport.mechanism}
                      </p>
                    </div>

                    {/* 诸派合参综述 */}
                    <div className="border border-[#D8C7A0] bg-[#FFFBF2] p-7 shadow-paper space-y-3">
                      <h4 className="font-bold text-sm tracking-widest text-[#27251F] border-b border-[#E6D7B5] pb-2">
                        【诸家学术论难】
                      </h4>
                      {finalReport.consensus?.length > 0 && (
                        <div className="text-xs space-y-1.5 pt-1">
                          <span className="font-bold text-[#385B62]">● 诸医共识：</span>
                          {finalReport.consensus.map((c, i) => (
                            <p key={i} className="text-[#594939] pl-3 leading-relaxed">- {c}</p>
                          ))}
                        </div>
                      )}
                      {finalReport.divergence?.length > 0 && (
                        <div className="text-xs space-y-1.5 pt-2">
                          <span className="font-bold text-[#9B4B3E]">▲ 学术分歧与鉴别：</span>
                          {finalReport.divergence.map((d, i) => (
                            <p key={i} className="text-[#594939] pl-3 leading-relaxed">- {d}</p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 右1列：四诊征象簿与宣教 */}
                  <div className="space-y-6">
                    {/* 四诊凭据 */}
                    <div className="border border-[#D8C7A0] bg-[#FAF3E3] p-6 shadow-paper space-y-3">
                      <h4 className="font-bold text-sm tracking-widest text-[#27251F] border-b border-[#E6D7B5] pb-2">
                        【四诊所据】
                      </h4>
                      <div className="space-y-2 max-h-64 overflow-y-auto pr-1 text-xs">
                        {finalReport.evidence?.map((e, i) => (
                          <div key={i} className="bg-white p-2.5 border-l-2 border-[#9B4B3E] border-[#EADBB8] text-[#554536]">
                            {e}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* 摄生宣教 */}
                    <div className="border border-[#D8C7A0] bg-[#FAF3E3] p-6 shadow-paper space-y-3">
                      <h4 className="font-bold text-sm tracking-widest text-[#27251F] border-b border-[#E6D7B5] pb-2">
                        【摄生调摄告诫】
                      </h4>
                      <div className="space-y-2 text-xs">
                        {finalReport.cautions?.map((c, i) => (
                          <div key={i} className="flex items-start gap-2 text-[#5A4B3C] leading-relaxed">
                            <span className="text-[#9B4B3E] font-bold">▪</span>
                            <span>{c}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <button
                        onClick={() => setView('selection')}
                        className="ink-button w-full justify-center text-xs tracking-widest !py-3 font-semibold"
                      >
                        <RefreshCw className="w-4 h-4" /> 重新启卷问诊
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                /* 五大学派各自主张册页 */
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {finalReport.panel?.map((op) => (
                    <div
                      key={op.school}
                      className="border border-[#D8C7A0] bg-[#FAF3E3] p-6 shadow-paper space-y-4 text-xs flex flex-col justify-between"
                    >
                      <div className="space-y-2.5">
                        <div className="flex items-center justify-between border-b border-[#E0D0B0] pb-2">
                          <span className="font-bold text-base text-[#27251F]">{op.title}</span>
                          <span className="seal !w-auto !h-auto px-2 py-0.5 !text-[10px]">
                            {op.source === 'llm' ? '深度推理' : '经典规则'}
                          </span>
                        </div>
                        <p className="text-[11px] text-[#7D6B55] tracking-wider">{op.name} 著札</p>
                        
                        <div className="bg-white p-3 border-l-2 border-[#9B4B3E] border-[#EADBB8]">
                          <span className="text-[10px] text-[#8C7A65] block">主张辨证：</span>
                          <span className="font-extrabold text-sm text-[#9B4B3E]">{op.diagnosis}</span>
                        </div>

                        <p className="text-[#594939] leading-relaxed line-clamp-5 pt-1">
                          {op.mechanism}
                        </p>
                      </div>

                      <div className="border-t border-[#E0D0B0] pt-3 space-y-1 font-serif">
                        <p className="font-semibold text-[#27251F]">治法：{op.treatment}</p>
                        <p className="text-[#9B4B3E] font-bold">方药：{op.formula || '随症斟酌'}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {/* 管理员管治枢纽 */}
          {view === 'admin' && (
            <motion.div
              key="admin"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-xl mx-auto w-full border-2 border-[#385B62] bg-[#FAF3E3] p-8 md:p-10 shadow-paper space-y-6"
            >
              <div className="flex items-center justify-between border-b border-[#D8C7A0] pb-4">
                <div className="flex items-center gap-3">
                  <div className="seal !w-8 !h-8 !border-[#385B62] !text-[#385B62]">枢</div>
                  <div>
                    <h3 className="text-xl font-bold tracking-wider text-[#27251F]">管治枢纽 · 模型接口配制</h3>
                    <p className="text-xs text-[#7A6854]">OpenAI 兼容双核路由设置 (运行时生效)</p>
                  </div>
                </div>
                <button
                  onClick={() => setView('home')}
                  className="text-xs px-3 py-1.5 border border-[#385B62] text-[#385B62] hover:bg-[#385B62] hover:text-white transition-all"
                >
                  返还前台
                </button>
              </div>

              <div className="space-y-4 text-xs font-sans">
                <div>
                  <label className="block text-[#594939] font-semibold mb-1">管理凭证 (Admin Token)</label>
                  <input
                    type="password"
                    value={adminToken}
                    onChange={(e) => setAdminToken(e.target.value)}
                    className="w-full p-2.5 border border-[#D8C7A0] bg-white font-mono"
                  />
                </div>

                <div>
                  <label className="block text-[#594939] font-semibold mb-1">兼容端点 (Base URL)</label>
                  <input
                    type="text"
                    value={adminBaseUrl}
                    onChange={(e) => setAdminBaseUrl(e.target.value)}
                    placeholder="https://vectide.cn/v1"
                    className="w-full p-2.5 border border-[#D8C7A0] bg-white font-mono"
                  />
                </div>

                <div>
                  <label className="block text-[#594939] font-semibold mb-1">API Key</label>
                  <input
                    type="password"
                    value={adminApiKey}
                    onChange={(e) => setAdminApiKey(e.target.value)}
                    placeholder="sk-..."
                    className="w-full p-2.5 border border-[#D8C7A0] bg-white font-mono"
                  />
                  {llmStatus?.api_key_masked && (
                    <span className="text-[10px] text-[#385B62] mt-1 block">
                      已持久化密钥：{llmStatus.api_key_masked}
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[#594939] font-semibold mb-1">
                      Fast 敏捷核 (追问/对话)
                    </label>
                    <input
                      type="text"
                      value={adminModelFast}
                      onChange={(e) => setAdminModelFast(e.target.value)}
                      placeholder="deepseek-v4-flash-0731"
                      className="w-full p-2.5 border border-[#D8C7A0] bg-white font-mono text-[11px]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#594939] font-semibold mb-1">
                      Pro 辨证核 (深层推理/整合)
                    </label>
                    <input
                      type="text"
                      value={adminModelPro}
                      onChange={(e) => setAdminModelPro(e.target.value)}
                      placeholder="deepseek-v4-pro-0813"
                      className="w-full p-2.5 border border-[#D8C7A0] bg-white font-mono text-[11px]"
                    />
                  </div>
                </div>
              </div>

              {adminMsg && (
                <div className="text-xs p-3 border border-[#D8C7A0] bg-[#FFFBF2] text-[#4F4030]">
                  {adminMsg}
                </div>
              )}

              <div className="flex gap-4 pt-2">
                <button
                  onClick={saveAdminSettings}
                  disabled={adminLoading}
                  className="ink-button flex-1 justify-center !py-2.5 text-xs font-semibold"
                >
                  保存并生效
                </button>
                <button
                  onClick={testAdminConnection}
                  disabled={adminLoading}
                  className="flex-1 border border-[#385B62] text-[#385B62] hover:bg-[#385B62] hover:text-white py-2.5 text-xs font-semibold transition-all disabled:opacity-50"
                >
                  测试连通性 (Both)
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* 底部印章与免责札 */}
      <footer className="border-t border-[#D8C7A0] py-3.5 text-center text-[11px] text-[#7A6854] bg-[#EFE4C8]/60 relative z-10 tracking-widest">
        本系统由多Agent协同中医辨证模型推演，案札仅供健康调摄与辨析参考，不可替代执业医师临证面诊。
      </footer>
    </div>
  )
}

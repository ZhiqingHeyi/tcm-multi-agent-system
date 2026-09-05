export type School = {
  id: string
  name: string
  title: string
  specialty: string
  style: string
  accent: string
}

export type Question = {
  id: string
  title: string
  key: string
  options: string[]
  optional?: boolean
}

export type QuestionModule = {
  module: string
  questions: Question[]
}

export type AgentOpinion = {
  school: string
  name: string
  title: string
  diagnosis: string
  confidence: number
  evidence: string[]
  counter: string[]
  mechanism: string
  treatment: string
  formula: string | null
  differentiation: string
  source: string
}

export type FinalReport = {
  engine: string
  diagnosis: string
  confidence: number
  mechanism: string
  treatment: string
  formula: string | null
  modifications: string
  consensus: string[]
  divergence: string[]
  cautions: string[]
  followup: string[]
  evidence: string[]
  risk_flags: string[]
  panel: AgentOpinion[]
}

export type FollowupResult = {
  questions: { title: string; key: string; options: string[] }[]
  ready: boolean
}

export type LLMStatus = {
  llm_base_url: string
  llm_model_fast: string
  llm_model_pro: string
  api_key_masked: string
  has_api_key: boolean
  configured: boolean
}

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8001'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

export const api = {
  agents: () => request<School[]>('/api/agents'),
  questionnaire: () => request<{ modules: QuestionModule[]; total: number }>('/api/questionnaire'),
  createConsultation: (school: string) =>
    request<{ id: string; school: string }>('/api/consultations', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ school }),
    }),
  saveAnswers: (id: string, facts: Record<string, string>) =>
    request<{ saved: number; risk_flags: string[] }>(`/api/consultations/${id}/answers`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ facts }),
    }),
  followup: (id: string) =>
    request<FollowupResult>(`/api/consultations/${id}/followup`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: '{}',
    }),
  llmStatus: () => request<LLMStatus>('/api/llm/status'),
  updateLLM: (cfg: { base_url?: string; api_key?: string; model_fast?: string; model_pro?: string }, token: string) =>
    request<LLMStatus>('/api/admin/llm', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-admin-token': token },
      body: JSON.stringify(cfg),
    }),
  testLLM: (token: string, role = 'both') =>
    request<{ ok: boolean; replies: Record<string, string> }>('/api/admin/llm/test', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-admin-token': token },
      body: JSON.stringify({ role }),
    }),
}

export async function streamReport(
  id: string,
  facts: Record<string, string>,
  handlers: {
    onStage?: (msg: string) => void
    onAgentResult?: (opinion: AgentOpinion) => void
    onReport?: (report: FinalReport) => void
    onError?: (err: Error) => void
  }
) {
  try {
    const res = await fetch(`${BASE}/api/consultations/${id}/report`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ facts }),
    })
    if (!res.ok) throw new Error(`HTTP error ${res.status}`)
    if (!res.body) throw new Error('Response body is null')

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() ?? ''

      for (const chunk of chunks) {
        if (!chunk.trim()) continue
        const lines = chunk.split('\n')
        let event = ''
        let dataStr = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) event = line.slice(7).trim()
          else if (line.startsWith('data: ')) dataStr = line.slice(6).trim()
        }
        if (!dataStr) continue
        try {
          const parsed = JSON.parse(dataStr)
          if (event === 'stage_started') handlers.onStage?.(parsed.message || '')
          else if (event === 'agent_result') handlers.onAgentResult?.(parsed as AgentOpinion)
          else if (event === 'report') handlers.onReport?.(parsed as FinalReport)
        } catch {
          // ignore chunk parse error
        }
      }
    }
  } catch (err: any) {
    handlers.onError?.(err instanceof Error ? err : new Error(String(err)))
  }
}

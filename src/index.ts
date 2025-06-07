import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { initializeApp } from 'firebase/app';
import { getFirestore, collection, addDoc, getDocs, DocumentData, DocumentReference, CollectionReference } from 'firebase/firestore';
import OpenAI from 'openai';
import path from 'path';

// Load environment variables from .env file
dotenv.config({path: './.env'});

const app = express();
const port = parseInt(process.env.PORT || '5002', 10);

// Middleware
app.use(cors());
app.use(express.json());

// Initialize OpenAI
if (!process.env.OPENAI_API_KEY) {
  console.error('OpenAI API Key is missing in environment variables');
  process.exit(1);
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize Firebase with environment variables
const firebaseConfig = {
  apiKey: process.env.FIREBASE_API_KEY,
  authDomain: process.env.FIREBASE_AUTH_DOMAIN,
  projectId: process.env.FIREBASE_PROJECT_ID,
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.FIREBASE_APP_ID,
  measurementId: process.env.FIREBASE_MEASUREMENT_ID
};

// Validate Firebase configuration
if (!process.env.FIREBASE_API_KEY || !process.env.FIREBASE_PROJECT_ID) {
  console.error('Required Firebase configuration is missing in environment variables');
  process.exit(1);
}

const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore(firebaseApp);

// Add retry utility function with more robust error handling
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const retryWithBackoff = async <T>(
  operation: () => Promise<T>,
  maxRetries: number = 8,
  initialDelay: number = 5000
): Promise<T> => {
  let retries = 0;
  let delay = initialDelay;
  let lastError: any;

  while (retries < maxRetries) {
    try {
      return await operation();
    } catch (error: any) {
      lastError = error;
      console.error(`Attempt ${retries + 1} failed:`, error.message);
      
      if (error.status === 503 || error.status === 429) {
        console.log(`Service unavailable or rate limited. Retrying in ${delay}ms...`);
        await sleep(delay);
        retries++;
        delay *= 2;
      } else {
        throw error;
      }
    }
  }
  
  throw new Error(`Failed after ${maxRetries} retries. Last error: ${lastError?.message}`);
};

// Add type definitions
type CareerPath = string;
type CareerPaths = Record<string, CareerPath[]>;

type CareerDetail = {
  title: string;
  averageSalary: string;
  requirements: {
    education: string[];
    certifications: string[];
    experience: string[];
  };
  description: string;
  relatedCompanies: string[];
  roleModels: string[];
};

type CareerDetails = Record<string, CareerDetail>;

// Fallback career paths for when AI service is unavailable
const fallbackCareerPaths: CareerPaths = {
  "Software Development": [
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "Mobile Developer",
    "DevOps Engineer"
  ],
  "Data Science": [
    "Data Analyst",
    "Machine Learning Engineer",
    "Data Engineer",
    "Business Intelligence Analyst",
    "Research Scientist"
  ],
  "Design": [
    "UI/UX Designer",
    "Graphic Designer",
    "Product Designer",
    "Motion Designer",
    "Interaction Designer"
  ],
  "Business": [
    "Product Manager",
    "Project Manager",
    "Business Analyst",
    "Marketing Manager",
    "Sales Manager"
  ]
};

// Fallback career details
const fallbackCareerDetails: CareerDetails = {
  "Software Developer": {
    title: "Software Developer",
    averageSalary: "$50K-80K entry, $80K-150K+ senior",
    requirements: {
      education: ["Bachelor's degree in Computer Science or related field"],
      certifications: ["AWS Certified Developer", "Microsoft Certified: Azure Developer Associate"],
      experience: ["2+ years of software development experience", "Experience with modern frameworks"]
    },
    description: "Software developers design, code, and maintain software applications and systems.",
    relatedCompanies: ["Google", "Microsoft", "Amazon", "Apple", "Meta"],
    roleModels: ["Linus Torvalds", "Guido van Rossum", "James Gosling"]
  }
};

// Simple MBTI guidance function
const getSimpleMBTIGuidance = (mbti: string): string => {
  const guidance: Record<string, string> = {
    'INTJ': 'strategic, analytical roles',
    'INFJ': 'meaningful, people-focused roles',
    'ENFP': 'creative, collaborative roles', 
    'ENTP': 'innovative, entrepreneurial roles',
    'ISTJ': 'structured, reliable roles',
    'ISFJ': 'supportive, service-oriented roles',
    'ISTP': 'hands-on, technical roles',
    'ISFP': 'creative, flexible roles',
    'INFP': 'values-driven, expressive roles',
    'INTP': 'research, analytical roles',
    'ESTP': 'dynamic, action-oriented roles',
    'ESFP': 'social, energetic roles',
    'ESTJ': 'leadership, management roles',
    'ESFJ': 'collaborative, caring roles',
    'ENFJ': 'mentoring, inspiring roles',
    'ENTJ': 'leadership, strategic roles'
  };
  return guidance[mbti?.toUpperCase()] || 'diverse career options';
};

// Helper: Extract JSON from OpenAI response
function extractJsonFromString(text: string | null | undefined): string | null {
  if (!text) return null;
  const trimmedText = text.trim();
  const firstCurly = trimmedText.indexOf('{');
  const firstSquare = trimmedText.indexOf('[');
  let startIndex = -1;
  if (firstCurly !== -1 && (firstSquare === -1 || firstCurly < firstSquare)) startIndex = firstCurly;
  else if (firstSquare !== -1) startIndex = firstSquare;
  if (startIndex === -1) return null;
  const lastCurly = trimmedText.lastIndexOf('}');
  const lastSquare = trimmedText.lastIndexOf(']');
  let endIndex = -1;
  if (lastCurly !== -1 && (lastSquare === -1 || lastCurly > lastSquare)) endIndex = lastCurly;
  else if (lastSquare !== -1) endIndex = lastSquare;
  if (endIndex === -1 || endIndex < startIndex) return null;
  return trimmedText.substring(startIndex, endIndex + 1);
}

// nodeContent 기반 fallback 함수
function getCareerSpecificSuggestions(nodeContent: string): string[] {
  const content = nodeContent.toLowerCase();
  
  // 직업별 맞춤 제안
  if (content.includes('developer') || content.includes('engineer')) {
    return [
      "Senior Software Engineer",
      "Technical Lead",
      "Full Stack Developer", 
      "DevOps Engineer",
      "Software Architect",
      "Backend Developer",
      "Frontend Developer",
      "Mobile App Developer"
    ];
  } else if (content.includes('designer') || content.includes('ux') || content.includes('ui')) {
    return [
      "Senior UX Designer",
      "Product Designer",
      "UI/UX Researcher",
      "Visual Designer",
      "Interaction Designer",
      "Design System Manager",
      "Creative Director",
      "Brand Designer"
    ];
  } else if (content.includes('manager') || content.includes('management')) {
    return [
      "Senior Product Manager",
      "Project Manager",
      "Program Manager",
      "Team Lead",
      "Operations Manager",
      "Strategy Manager",
      "Business Development Manager",
      "Marketing Manager"
    ];
  } else if (content.includes('data') || content.includes('analyst')) {
    return [
      "Senior Data Scientist",
      "Data Engineer",
      "Business Intelligence Analyst",
      "Machine Learning Engineer",
      "Data Analyst",
      "Research Scientist",
      "AI Engineer",
      "Analytics Manager"
    ];
  } else {
    // 일반적인 커리어 확장
    return [
      `Senior ${nodeContent}`,
      `${nodeContent} Lead`,
      `${nodeContent} Manager`,
      `${nodeContent} Specialist`,
      `${nodeContent} Consultant`,
      `${nodeContent} Director`,
      `${nodeContent} Expert`,
      `Principal ${nodeContent}`
    ];
  }
}

// Routes
app.get('/', (req, res) => {
  res.send('GrowGraph API is running');
});

// Generate mind map from user input
app.post('/api/generate-mindmap', async (req, res) => {
  try {
    const userData = req.body;
    console.log('Received user data:', userData);

    // 사용자가 원하는 경우 중앙 노드만 반환하거나, 전체 마인드맵 생성
    if (userData.centerOnly) {
      res.json({
        nodes: [
          { 
            id: 'root', 
            data: { label: userData.jobPath || userData.aim || 'Career Center' },
            position: { x: 0, y: 0 } 
          }
        ],
        edges: []
      });
      return;
    }

    const prompt = `Create a career mind map for someone with:
- Career Goal: ${userData.aim}
- Job Path: ${userData.jobPath}
- Interests: ${userData.hobby}
- MBTI: ${userData.mbti} (prefer ${getSimpleMBTIGuidance(userData.mbti)})
- Target Salary: ${userData.salary}
- Role Model: ${userData.roleModel}

Generate 6-8 specific job titles IN KOREAN with time estimates that match their goals and MBTI preferences. 

Return JSON format:
{
  "nodes": [
    { "id": "1", "data": { "label": "[Main Career Goal in Korean]" }, "position": { "x": 0, "y": 0 } },
    { "id": "2", "data": { "label": "[Korean Job Title] (경력 X년)" }, "position": { "x": -200, "y": -150 } }
    // ... 5-7 more job nodes around the center
  ],
  "edges": [
    { "id": "e1-2", "source": "1", "target": "2" }
    // ... edges connecting center to each job
  ]
}

IMPORTANT: 
- All job titles must be in Korean with proper nouns (company names, people names, technologies) in English
- Add time estimate in parentheses: "(신입)", "(경력 2-3년)", "(경력 5년+)", "(경력 10년+)" 
- Examples: "시니어 Product Manager (경력 5년+)", "UX 리서처 (경력 2-3년)", "Google 소프트웨어 엔지니어 (경력 3년+)"
- Use specific Korean job titles with English proper nouns where appropriate`;


    try {
      const result = await retryWithBackoff(async () => {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: "You generate career mind maps in JSON format with specific job titles."
            },
            {
              role: "user",
              content: prompt
            }
          ],
          temperature: 0.7,
          max_tokens: 1024
        });
        const content = completion.choices[0]?.message?.content;
        if (!content) {
          throw new Error('No content received from OpenAI');
        }
        return content;
      });

      const cleanedJsonString = extractJsonFromString(result);
      if (!cleanedJsonString) {
        throw new Error('Failed to extract valid JSON from AI response');
      }

      const mindMap = JSON.parse(cleanedJsonString);
      // 노드 data 보정: data가 없거나 label이 없으면 최대한 다양한 경우를 보정
      if (Array.isArray(mindMap.nodes)) {
        mindMap.nodes = mindMap.nodes.map((node: any, idx: number) => {
          let label = node.id;
          // 1. data가 객체이고 label이 있으면 label 사용
          if (node.data && typeof node.data === 'object' && 'label' in node.data && node.data.label) {
            label = node.data.label;
          }
          // 2. data가 문자열이면 label로 사용
          else if (typeof node.data === 'string' && node.data) {
            label = node.data;
          }
          // 3. data가 배열이고 첫 번째 값이 문자열이면 label로 사용
          else if (Array.isArray(node.data) && typeof node.data[0] === 'string') {
            label = node.data[0];
          }
          // 4. name, title, text, value 등의 필드가 있으면 label로 사용
          else if (node.data && typeof node.data === 'object') {
            if (node.data.name) label = node.data.name;
            else if (node.data.title) label = node.data.title;
            else if (node.data.text) label = node.data.text;
            else if (node.data.value) label = node.data.value;
          }
          // 5. id가 있으면 id 사용, 아니면 fallback
          else if (node.id) {
            label = node.id;
          } else {
            label = 'No Label';
          }
          // 디버깅용 상세 로그
          console.log(`[SERVER] node[${idx}] id:`, node.id, 'data:', JSON.stringify(node.data), 'label:', label);
          return {
            ...node,
            data: { label }
          };
        });
      }
      // 전체 mindMap 구조도 출력
      console.log('[SERVER] 최종 mindMap:', JSON.stringify(mindMap, null, 2));
      res.json(mindMap);
    } catch (aiError) {
      console.error('AI service failed, using fallback:', aiError);
      // Use fallback mind map
      const fallbackMindMap = {
        nodes: [
          { id: "1", data: { label: userData.aim || "Career Exploration" }, position: { x: 0, y: 0 } },
          { id: "2", data: { label: "Software Development" }, position: { x: -200, y: 100 } },
          { id: "3", data: { label: "Data Science" }, position: { x: 200, y: 100 } },
          { id: "4", data: { label: "Design" }, position: { x: -200, y: -100 } },
          { id: "5", data: { label: "Business" }, position: { x: 200, y: -100 } }
        ],
        edges: [
          { id: "e1-2", source: "1", target: "2" },
          { id: "e1-3", source: "1", target: "3" },
          { id: "e1-4", source: "1", target: "4" },
          { id: "e1-5", source: "1", target: "5" }
        ]
      };
      res.json(fallbackMindMap);
    }
  } catch (error: any) {
    console.error('Failed to generate mind map:', error);
    res.status(500).json({ 
      error: 'Failed to generate mind map',
      details: error.message,
      status: error.status
    });
  }
});

// Save mind map
app.post('/api/mindmap', async (req, res) => {
  try {
    const { nodes, edges } = req.body;
    const docRef = await addDoc(collection(db, 'mindmaps'), {
      nodes,
      edges,
      createdAt: new Date().toISOString()
    });
    res.json({ id: docRef.id });
  } catch (error) {
    console.error('Failed to save mind map:', error);
    res.status(500).json({ error: 'Failed to save mind map' });
  }
});

// Get mind maps
app.get('/api/mindmap', async (req, res) => {
  try {
    const querySnapshot = await getDocs(collection(db, 'mindmaps'));
    const mindmaps = querySnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));
    res.json(mindmaps);
  } catch (error) {
    console.error('Failed to fetch mind maps:', error);
    res.status(500).json({ error: 'Failed to fetch mind maps' });
  }
});

// Get AI suggestions
app.post('/api/suggestions', async (req, res) => {
  try {
    const { nodeContent } = req.body;
    
    const prompt = `Expand "${nodeContent}" into 8-12 specific related job titles or career paths IN KOREAN with time estimates. 

Return only a JSON array of Korean strings with English proper nouns and time estimates.

Format: ["Korean Job Title (경력 X년)", "Another Korean Job Title (신입)", ...]

Example: ["시니어 Software Engineer (경력 5년+)", "Google Product Manager (경력 3-5년)", "스타트업 CTO (경력 10년+)"]

IMPORTANT: Keep company names, technologies, and proper nouns in English within Korean job titles.`;
    
    try {
      const completion = await openai.chat.completions.create({
        messages: [
          {
            role: "system", 
            content: "You expand career terms into specific job titles. Return JSON arrays only."
          },
          { 
            role: "user", 
            content: prompt 
          }
        ],
        model: "gpt-4o",
        temperature: 0.7,
        max_tokens: 512
      });
      
      const responseContent = completion.choices[0].message.content;
      const cleanedJsonString = extractJsonFromString(responseContent);
      
      if (!cleanedJsonString) {
        throw new Error('Failed to extract valid JSON from AI response');
      }
      
      const suggestions = JSON.parse(cleanedJsonString);
      res.json({ suggestions });
    } catch (aiError) {
      console.error('AI service failed, using career-specific fallback:', aiError);
      const fallbackSuggestions = getCareerSpecificSuggestions(nodeContent);
      res.json({ suggestions: fallbackSuggestions });
    }
  } catch (error) {
    console.error('Failed to generate suggestions:', error);
    res.status(500).json({ error: 'Failed to generate suggestions' });
  }
});

// Get career details
app.post('/api/career-details', async (req, res) => {
  try {
    const { careerTitle } = req.body;
    
    const prompt = `Career info for "${careerTitle}" in JSON with Korean content:

{
  "title": "${careerTitle}",
  "averageSalary": "한국 기준 연봉 (신입: X만원, 경력: X만원, 시니어: X만원)",
  "requirements": {
    "education": ["한국어로 학력 요구사항"],
    "certifications": ["한국어로 자격증/기술 요구사항 (영어 기술명 유지)"],
    "experience": ["한국어로 경력 요구사항"]
  },
  "description": "한국어로 직무 설명",
  "relatedCompanies": ["Major companies hiring this role"],
  "roleModels": ["Notable professionals with Korean description"],
  "timeToReach": {
    "신입": "경력 0년",
    "주니어": "경력 1-3년", 
    "시니어": "경력 5-7년",
    "리드": "경력 8년+"
  }
}

IMPORTANT: 
- Provide all content in Korean except company names, people names, and technology names
- Add realistic time estimates for career progression
- Include Korean salary information
- Keep proper nouns (Apple, Google, React, Python, etc.) in English`;
    
    try {
      const result = await retryWithBackoff(async () => {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: "You provide concise career information in JSON format."
            },
            {
              role: "user",
              content: prompt
            }
          ],
          temperature: 0.7,
          max_tokens: 800
        });
        const content = completion.choices[0]?.message?.content;
        if (!content) {
          throw new Error('No content received from OpenAI');
        }
        return content;
      });

      const cleanedJsonString = extractJsonFromString(result);
      if (!cleanedJsonString) {
        throw new Error('Failed to extract valid JSON from AI response');
      }

      const careerInfo = JSON.parse(cleanedJsonString);
      console.log('Successfully parsed career info:', careerInfo);
      res.json(careerInfo);
    } catch (aiError) {
      console.error('AI service failed, using fallback:', aiError);
      
      // 글로벌 기준 간단한 fallback
      const fallbackInfo = {
        title: careerTitle,
        averageSalary: "$50K-80K entry, $80K-150K+ senior",
        requirements: {
          education: ["Bachelor's degree preferred"],
          certifications: ["Industry-standard certifications"],
          experience: ["2+ years relevant experience"]
        },
        description: `${careerTitle} professionals solve problems using specialized skills and knowledge.`,
        relatedCompanies: ["Google", "Microsoft", "Apple", "Amazon", "Meta"],
        roleModels: ["Industry leaders", "Successful practitioners"]
      };
      res.json(fallbackInfo);
    }
  } catch (error: any) {
    console.error('Failed to generate career details:', error);
    res.status(500).json({ 
      error: 'Failed to generate career details',
      details: error.message,
      status: error.status
    });
  }
});

// Expand career node
app.post('/api/expand-career', async (req, res) => {
  try {
    const { careerTitle, level } = req.body;
    
    const prompt = `Given the career "${careerTitle}", generate ${level === 1 ? 'main career paths' : 'specific roles and specializations'} in this field.
    Format the response as a JSON array of strings, where each string is a career path or role.
    Return ONLY the JSON array, no other text.`;
    
    try {
      const result = await retryWithBackoff(async () => {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: "You are a helpful assistant that generates career paths in JSON format."
            },
            {
              role: "user",
              content: prompt
            }
          ],
          temperature: 0.7,
          max_tokens: 1024
        });
        const content = completion.choices[0]?.message?.content;
        if (!content) {
          throw new Error('No content received from OpenAI');
        }
        return content;
      });

      const cleanedJsonString = extractJsonFromString(result);
      if (!cleanedJsonString) {
        throw new Error('Failed to extract valid JSON from AI response');
      }

      const careerPaths = JSON.parse(cleanedJsonString);
      console.log('Successfully parsed career paths:', careerPaths);
      res.json({ careerPaths });
    } catch (aiError) {
      console.error('AI service failed, using fallback:', aiError);
      // Use fallback career paths
      const fallbackPaths = fallbackCareerPaths[careerTitle as keyof CareerPaths] || 
        ["Senior " + careerTitle, "Lead " + careerTitle, "Principal " + careerTitle];
      res.json({ careerPaths: fallbackPaths });
    }
  } catch (error: any) {
    console.error('Failed to expand career node:', error);
    res.status(500).json({ 
      error: 'Failed to expand career node',
      details: error.message,
      status: error.status
    });
  }
});

// Start server with port fallback
const startServer = async (port: number) => {
  try {
    app.listen(port, () => {
      console.log(`Server is running on port ${port}`);
    });
  } catch (error: any) {
    if (error.code === 'EADDRINUSE') {
      console.log(`Port ${port} is busy, trying ${port + 1}`);
      await startServer(port + 1);
    } else {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
  }
};

// Start the server
startServer(port).catch(error => {
  console.error('Failed to start server:', error);
  process.exit(1);
});
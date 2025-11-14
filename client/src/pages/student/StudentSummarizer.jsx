import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { FileText, Upload, MessageCircle, Send, BookOpen, ChevronRight, ChevronDown } from 'lucide-react';
import api from '../../lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import Button from '../../components/ui/Button';
import toast from 'react-hot-toast';

export default function StudentSummarizer() {
  // PDF Summarizer states
  const [pdfSummary, setPdfSummary] = useState('');
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfFileName, setPdfFileName] = useState('');
  const [pdfRawText, setPdfRawText] = useState('');
  const [contentOverview, setContentOverview] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [expandedSections, setExpandedSections] = useState(new Set());
  const [summarizingSection, setSummarizingSection] = useState(null);
  const [questionsAndAnswers, setQuestionsAndAnswers] = useState([]);
  const fileInputRef = useRef(null);
  
  // Q&A states
  const [question, setQuestion] = useState('');
  const [qaLoading, setQaLoading] = useState(false);
  const [qaHistory, setQaHistory] = useState([]);

  // Handle PDF file selection
  const handlePdfUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === 'application/pdf') {
        setPdfFileName(file.name);
        toast.success('File selected! Click "Summarize PDF" to process.');
      } else {
        toast.error('Please upload a PDF file');
        e.target.value = '';
      }
    }
  };

  // Handle PDF summarization
  const handlePdfSummarize = async () => {
    if (!fileInputRef.current?.files?.[0]) {
      toast.error('Please select a file first');
      return;
    }

    setPdfLoading(true);
    setPdfSummary('');
    setPdfRawText('');
    setContentOverview([]);
    setStatistics(null);
    setExpandedSections(new Set());
    setQaHistory([]);
    setQuestionsAndAnswers([]);

    try {
      const file = fileInputRef.current.files[0];
      const formData = new FormData();
      formData.append('document', file);

      // Use the new summarize-pdf endpoint
      const { data } = await api.post('/api/ai/summarize-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes timeout
      });

      // ALWAYS SET SUMMARY - even if empty or error
      if (data.summary) {
        setPdfSummary(data.summary);
      } else {
        // If no summary, set a default message
        setPdfSummary('Summary is being generated. Please wait...');
      }

      if (data.error) {
        toast.error(data.error);
        // Still show summary if available, otherwise show error
        if (!data.summary) {
          setPdfSummary(`Error: ${data.error}. Please try again.`);
        }
        if (!data.summary) {
          return;
        }
      }

      if (!data.success && !data.summary) {
        toast.error(data.error || 'Failed to process PDF');
        setPdfSummary('Error: Failed to process PDF. Please try again.');
        return;
      }

      // Store questions and answers (optional)
      if (data.questions_and_answers && Array.isArray(data.questions_and_answers)) {
        setQuestionsAndAnswers(data.questions_and_answers);
        if (data.questions_and_answers.length > 0) {
          toast.success('PDF summarized successfully with questions and answers!');
        } else {
          toast.success('PDF summarized successfully!');
        }
      } else {
        toast.success('PDF summarized successfully!');
      }
    } catch (error) {
      console.error('PDF summarization error:', error);
      const errorMessage = error.response?.data?.error || error.message || 'Failed to summarize PDF. Please check if AI service is running.';
      
      toast.error(errorMessage);
      
      // ALWAYS SET SUMMARY - even on error
      if (error.response?.data?.summary) {
        setPdfSummary(error.response.data.summary);
      } else {
        setPdfSummary(`Error: ${errorMessage}. Please try again or check if the AI service is running.`);
      }
      
      // Set Q&A if available even in error
      if (error.response?.data?.questions_and_answers) {
        setQuestionsAndAnswers(error.response.data.questions_and_answers);
      }
    } finally {
      setPdfLoading(false);
    }
  };

  // Handle section summarization
  const handleSummarizeSection = async (sectionText, sectionType = 'heading') => {
    if (!fileInputRef.current?.files?.[0]) {
      toast.error('Please select a file first');
      return;
    }

    setSummarizingSection(sectionText);
    
    try {
      const file = fileInputRef.current.files[0];
      // For now, we'll use the existing summary data
      // In a full implementation, you'd call the /api/summarize-section endpoint
      toast.info('Section summarization feature coming soon');
    } catch (error) {
      console.error('Section summarization error:', error);
      toast.error('Failed to summarize section');
    } finally {
      setSummarizingSection(null);
    }
  };

  // Toggle section expansion
  const toggleSection = (index) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSections(newExpanded);
  };

  // Handle Q&A based on PDF content (searches exact PDF lines)
  const handleAskQuestion = async () => {
    if (!question.trim()) {
      toast.error('Please enter a question');
      return;
    }

    if (!fileInputRef.current?.files?.[0]) {
      toast.error('Please upload a PDF file first');
      return;
    }

    setQaLoading(true);
    const currentQuestion = question;
    setQuestion('');

    try {
      // Add question to history
      const newQaHistory = [...qaHistory, { 
        question: currentQuestion, 
        answer: '...',
        sources: []
      }];
      setQaHistory(newQaHistory);

      // Use the new PDF Q&A endpoint that searches through PDF content
      const file = fileInputRef.current.files[0];
      const formData = new FormData();
      formData.append('document', file);
      formData.append('question', currentQuestion);

      const { data } = await api.post('/api/ai/pdf-qa', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (data.error) {
        throw new Error(data.error);
      }

      const answerText = data.answer || 'I could not generate an answer. Please try again.';
      const sources = data.sources || [];
      
      // Update the last Q&A entry with the answer
      const updatedQaHistory = [...newQaHistory];
      updatedQaHistory[updatedQaHistory.length - 1].answer = answerText;
      updatedQaHistory[updatedQaHistory.length - 1].sources = sources;
      setQaHistory(updatedQaHistory);
      toast.success('Answer generated from PDF content!');
    } catch (error) {
      console.error('Q&A error:', error);
      toast.error(error.response?.data?.error || 'Failed to get answer. Please try again.');
      
      // Update the last Q&A entry with error
      const updatedQaHistory = [...qaHistory];
      if (updatedQaHistory.length > 0) {
        updatedQaHistory[updatedQaHistory.length - 1].answer = 'Error: Could not generate answer. Please try again.';
        setQaHistory(updatedQaHistory);
      }
    } finally {
      setQaLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-5xl font-bold text-gray-900 dark:text-white">
            PDF Summarizer
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Upload a PDF document and get an AI-powered summary with structured content overview
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column - Upload Section */}
          <div className="lg:col-span-1">
            <Card className="sticky top-6">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Upload className="text-blue-500 dark:text-blue-400" size={24} />
                  <span>Upload PDF</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">
                    Select PDF Document
                  </label>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handlePdfUpload}
                    className="hidden"
                  />
                  <div className="space-y-3">
                    <Button
                      variant="secondary"
                      className="w-full"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={pdfLoading}
                    >
                      <Upload size={18} className="mr-2" />
                      {pdfFileName || 'Choose PDF File'}
                    </Button>
                    {pdfFileName && (
                      <div className="rounded-lg bg-gray-50 dark:bg-gray-800 p-3">
                        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                          {pdfFileName}
                        </p>
                        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Ready to process
                        </p>
                      </div>
                    )}
                    {pdfFileName && (
                      <Button
                        variant="primary"
                        className="w-full"
                        onClick={handlePdfSummarize}
                        disabled={pdfLoading}
                      >
                        {pdfLoading ? (
                          <>
                            <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                            Processing...
                          </>
                        ) : (
                          <>
                            <FileText size={18} className="mr-2" />
                            Summarize PDF
                          </>
                        )}
                      </Button>
                    )}
                  </div>
                </div>

                {/* Statistics */}
                {statistics && (
                  <div className="mt-6 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 p-4">
                    <h3 className="mb-3 text-sm font-semibold text-gray-900 dark:text-gray-100">
                      Document Statistics
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Total Headings:</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {statistics.total_headings || 0}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Total Lines:</span>
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {statistics.total_lines || 0}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Main Summary - ALWAYS SHOW THIS */}
            <Card className="shadow-lg">
              <CardHeader className="bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20">
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="text-green-600 dark:text-green-400" size={24} />
                  <span>Document Summary</span>
                </CardTitle>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                  AI-generated summary of the PDF content
                </p>
              </CardHeader>
              <CardContent className="pt-6">
                {pdfLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="flex flex-col items-center space-y-4">
                      <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Processing PDF and generating summary...
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500">
                        This may take a few moments
                      </p>
                    </div>
                  </div>
                ) : pdfSummary ? (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="prose prose-lg dark:prose-invert max-w-none"
                  >
                    <div 
                      className="text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap"
                    >
                      {pdfSummary.split('\n').map((paragraph, idx) => {
                        if (!paragraph.trim()) return <br key={idx} />;
                        
                        // Check for headings
                        if (paragraph.startsWith('## ')) {
                          return (
                            <h2 key={idx} className="text-2xl font-bold mt-8 mb-4 text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700 pb-2">
                              {paragraph.replace(/^## /, '')}
                            </h2>
                          );
                        }
                        if (paragraph.startsWith('### ')) {
                          return (
                            <h3 key={idx} className="text-xl font-semibold mt-6 mb-3 text-gray-800 dark:text-gray-200">
                              {paragraph.replace(/^### /, '')}
                            </h3>
                          );
                        }
                        
                        // Check for bullet points
                        if (paragraph.trim().startsWith('* ') || paragraph.trim().startsWith('- ')) {
                          return (
                            <li key={idx} className="ml-6 list-disc mb-2 text-gray-700 dark:text-gray-300">
                              {paragraph.replace(/^[\*\-\s]+/, '')}
                            </li>
                          );
                        }
                        
                        // Regular paragraph
                        return (
                          <p key={idx} className="mb-4 text-gray-700 dark:text-gray-300 leading-relaxed">
                            {paragraph.split(/\*\*(.*?)\*\*/g).map((part, partIdx) => {
                              if (partIdx % 2 === 1) {
                                return <strong key={partIdx} className="font-semibold text-gray-900 dark:text-gray-100">{part}</strong>;
                              }
                              return <span key={partIdx}>{part}</span>;
                            })}
                          </p>
                        );
                      })}
                    </div>
                  </motion.div>
                ) : (
                  <div className="text-center py-12">
                    <FileText size={48} className="mx-auto mb-4 text-gray-400 dark:text-gray-500" />
                    <p className="text-gray-600 dark:text-gray-400">
                      Upload a PDF and click "Summarize PDF" to generate a summary
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Questions and Answers Section */}
            {questionsAndAnswers && questionsAndAnswers.length > 0 && (
              <Card className="shadow-lg">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
                  <CardTitle className="flex items-center space-x-2">
                    <MessageCircle className="text-blue-600 dark:text-blue-400" size={24} />
                    <span>Questions & Answers</span>
                  </CardTitle>
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                    Important questions and their answers based on the PDF content
                  </p>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-6">
                    {questionsAndAnswers.map((qa, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="space-y-3"
                      >
                        <div className="rounded-lg bg-blue-50 dark:bg-blue-900/20 p-4 border-l-4 border-blue-500">
                          <div className="flex items-start space-x-2">
                            <MessageCircle className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
                            <div className="flex-1">
                              <p className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-1">
                                Question {index + 1}:
                              </p>
                              <p className="text-sm text-blue-800 dark:text-blue-200">{qa.question}</p>
                            </div>
                          </div>
                        </div>
                        <div className="rounded-lg bg-white dark:bg-gray-800 p-4 ml-6 border border-gray-200 dark:border-gray-700 shadow-sm">
                          <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                            Answer:
                          </p>
                          <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                            <div className="prose prose-sm dark:prose-invert max-w-none">
                              <p className="whitespace-pre-wrap leading-relaxed">{qa.answer}</p>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Content Overview - Structured Sections */}
            {contentOverview && contentOverview.length > 0 && (
              <Card className="shadow-lg">
                <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20">
                  <CardTitle className="flex items-center space-x-2">
                    <BookOpen className="text-purple-600 dark:text-purple-400" size={24} />
                    <span>Document Structure</span>
                  </CardTitle>
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                    Explore the PDF's hierarchical structure with headings and subheadings
                  </p>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-3">
                    {contentOverview.map((section, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden shadow-sm hover:shadow-md transition-shadow"
                      >
                        <button
                          onClick={() => toggleSection(index)}
                          className="w-full flex items-center justify-between p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors group"
                        >
                          <div className="flex items-center space-x-3 flex-1 text-left">
                            <div className="flex-shrink-0">
                              {expandedSections.has(index) ? (
                                <ChevronDown className="h-5 w-5 text-purple-500 dark:text-purple-400" />
                              ) : (
                                <ChevronRight className="h-5 w-5 text-gray-400 dark:text-gray-500 group-hover:text-purple-500 dark:group-hover:text-purple-400 transition-colors" />
                              )}
                            </div>
                            <div className="flex-1">
                              <h3 className="font-semibold text-gray-900 dark:text-gray-100 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
                                {section.heading}
                              </h3>
                              {section.subheadings && section.subheadings.length > 0 && (
                                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                  {section.subheadings.length} subheading{section.subheadings.length !== 1 ? 's' : ''}
                                </p>
                              )}
                            </div>
                          </div>
                        </button>
                        
                        {expandedSections.has(index) && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="border-t border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/50"
                          >
                            <div className="p-4 space-y-3">
                              {section.subheadings && section.subheadings.length > 0 && (
                                <div className="space-y-2">
                                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                                    <BookOpen className="h-4 w-4 mr-2 text-purple-500" />
                                    Subheadings:
                                  </h4>
                                  {section.subheadings.map((subheading, subIndex) => {
                                    // Handle both string and object formats
                                    const subheadingText = typeof subheading === 'string' 
                                      ? subheading 
                                      : subheading?.subheading || subheading?.text || 'Unknown subheading';
                                    return (
                                      <div
                                        key={subIndex}
                                        className="pl-4 py-2.5 rounded-md bg-white dark:bg-gray-800 border-l-4 border-purple-400 dark:border-purple-600 shadow-sm hover:shadow transition-shadow"
                                      >
                                        <p className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                                          {subheadingText}
                                        </p>
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                              {section.content_preview && (
                                <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                                  <p className="text-sm text-gray-600 dark:text-gray-400 italic leading-relaxed">
                                    {section.content_preview}
                                  </p>
                                </div>
                              )}
                            </div>
                          </motion.div>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Q&A Section */}
            {pdfFileName && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <MessageCircle className="text-blue-500 dark:text-blue-400" size={24} />
                    <span>Ask Questions About the PDF</span>
                  </CardTitle>
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                    Get answers directly from the PDF content with source references
                  </p>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Q&A History */}
                  {qaHistory.length > 0 && (
                    <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
                      {qaHistory.map((qa, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="space-y-3"
                        >
                          <div className="rounded-lg bg-blue-50 dark:bg-blue-900/20 p-4 border-l-4 border-blue-500">
                            <div className="flex items-start space-x-2">
                              <MessageCircle className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
                              <div className="flex-1">
                                <p className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-1">
                                  Question:
                                </p>
                                <p className="text-sm text-blue-800 dark:text-blue-200">{qa.question}</p>
                              </div>
                            </div>
                          </div>
                          <div className="rounded-lg bg-white dark:bg-gray-800 p-4 ml-6 border border-gray-200 dark:border-gray-700 shadow-sm">
                            <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                              Answer:
                            </p>
                            <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                              {qa.answer === '...' ? (
                                <div className="flex items-center gap-2 text-gray-500">
                                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
                                  <span>Searching PDF content and generating answer...</span>
                                </div>
                              ) : (
                                <>
                                  <div className="prose prose-sm dark:prose-invert max-w-none">
                                    <p className="whitespace-pre-wrap leading-relaxed">{qa.answer}</p>
                                  </div>
                                  {qa.sources && qa.sources.length > 0 && (
                                    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                                      <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">
                                        Sources:
                                      </p>
                                      <div className="flex flex-wrap gap-2">
                                        {qa.sources.map((source, idx) => (
                                          <span
                                            key={idx}
                                            className="inline-flex items-center px-2 py-1 rounded-md bg-gray-100 dark:bg-gray-700 text-xs font-medium text-gray-700 dark:text-gray-300"
                                          >
                                            {source}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {/* Question Input */}
                  <div className="flex gap-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex-1 relative">
                      <input
                        type="text"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && !qaLoading && handleAskQuestion()}
                        placeholder="Ask a question about the PDF content (e.g., 'What is mentioned about X?', 'Explain Y')"
                        className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-4 py-3 pr-12 text-black dark:text-white outline-none focus:border-blue-500 dark:focus:border-blue-400 focus:ring-2 focus:ring-blue-200 dark:focus:ring-blue-800 transition-all"
                        disabled={qaLoading}
                      />
                      {question && (
                        <button
                          onClick={() => setQuestion('')}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                        >
                          ×
                        </button>
                      )}
                    </div>
                    <Button
                      variant="primary"
                      onClick={handleAskQuestion}
                      disabled={qaLoading || !question.trim()}
                      className="px-6"
                    >
                      {qaLoading ? (
                        <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      ) : (
                        <>
                          <Send size={18} className="mr-2" />
                          Ask
                        </>
                      )}
                    </Button>
                  </div>
                  {!pdfSummary && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 italic">
                      💡 Tip: You can ask questions directly without summarizing first. The system will search through the PDF content to find answers.
                    </p>
                  )}
                </CardContent>
              </Card>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}

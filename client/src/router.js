import Vue from 'vue'
import Router from 'vue-router'
import Upload from '@/views/Upload'
import Result from '@/views/Result'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/',
      name: 'Upload',
      component: Upload
    },
    {
        path: '/result',
        name: 'result',
        component: Result
    }
  ]
})
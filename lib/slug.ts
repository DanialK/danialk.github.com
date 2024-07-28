import { slug as slugger } from 'github-slugger'

export default function slug(value, maintainCase?: boolean | undefined) {
  switch (value) {
    case 'C#':
      return 'csharp'
    default:
      return slugger(value, maintainCase)
  }
}
